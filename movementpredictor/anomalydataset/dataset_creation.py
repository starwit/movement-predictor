import os
import io
import csv
import json
import time
import tarfile
import pybase64
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from visionapi.sae_pb2 import SaeMessage
from visionlib import saedump

# import your tracking utilities
from movementpredictor.data import datamanagement, datafilterer  

from movementpredictor.config import ModelConfig
config = ModelConfig()



def atomic_write_bytes(path, data):
    """Write bytes atomically to avoid partial files on crash."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class ShardWriter:
    """
    Minimal WebDataset-style shard writer (.tar, uncompressed).
    Writes <key>.jpg entries (no per-sample JSON; global CSV/JSONL handles metadata).
    """
    def __init__(
        self,
        out_dir: str,
        prefix: str = "frames",
        max_shard_size_bytes: int = 1_000_000_000,  # ~1 GB
        max_samples_per_shard: int = 10000,
        flush_every: int = 64,
    ):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.prefix = prefix
        self.max_size = max_shard_size_bytes
        self.max_samples = max_samples_per_shard
        self.flush_every = flush_every

        self.shard_idx = 0
        self.samples_in_shard = 0
        self.bytes_in_shard = 0
        self.tar = None
        self._fobj = None
        self.current_path = None

        self._open_new_shard()

    def _open_new_shard(self):
        self._close_current()
        self.current_path = os.path.join(self.out_dir, f"{self.prefix}-{self.shard_idx:06d}.tar")
        self._fobj = open(self.current_path, "wb")
        self.tar = tarfile.open(fileobj=self._fobj, mode="w")
        self.samples_in_shard = 0
        self.bytes_in_shard = 0

    def _close_current(self):
        if self.tar is not None:
            self.tar.close()
            self._fobj.flush()
            os.fsync(self._fobj.fileno())
            self._fobj.close()
            self.tar = None
            self._fobj = None

    def add(self, key: str, jpg_bytes: bytes):
        now = int(time.time())
        ti = tarfile.TarInfo(name=f"{key}.jpg")
        ti.size = len(jpg_bytes)
        ti.mtime = now
        self.tar.addfile(ti, io.BytesIO(jpg_bytes))
        # account for data + tar header/block overhead (roughly)
        self.bytes_in_shard += ti.size + 512
        self.samples_in_shard += 1

        # periodic flush
        if self.samples_in_shard % self.flush_every == 0:
            self.tar.fileobj.flush()
            os.fsync(self.tar.fileobj.fileno())

        # rollover if limits reached
        if (self.samples_in_shard >= self.max_samples) or (self.bytes_in_shard >= self.max_size):
            self.shard_idx += 1
            self._open_new_shard()

    def close(self):
        self._close_current()



def create_raw_dataset(
    paths_sae_dumps: str,
    path_store: str,
    *,
    max_frames: int | None = None,
    flush_every: int = 64,
    workers: int = 2,
    write_mode: str = "shards",        # "files" | "shards" | "both"
    shard_prefix: str = "frames",
    max_shard_size_bytes: int = 1_000_000_000,
    max_samples_per_shard: int = 10000
) -> None:
    """
    Single pass over SAE dump that:
      - writes frames (files and/or shards),
      - writes an index.csv for frames,
      - writes detections with extra mapping fields: frame_index, frame_key, shard.

    Detections output:
      - "jsonl": object_detections.jsonl (one record per line) -> memory safe
      - "json" : object_detections.json  (single array) -> may use lots of RAM
    """
    os.makedirs(path_store, exist_ok=True)
    max_frames_per_dump = int(max_frames/len(paths_sae_dumps)) if max_frames is not None else None

    # --- tracking lookup (bbox by (object_id, capture_ts)) ---
    track_manager = datamanagement.TrackingDataManager()
    lookup_dict = {}  # accumulate across all dumps

    for path_sae_dump in paths_sae_dumps:
        tracked_objects = track_manager.getTrackedBaseData(path_sae_dump, inferencing=False, max_frames=max_frames_per_dump)
        tracked_objects = datafilterer.DataFilterer().apply_filtering(tracked_objects)

        # build per-file mapping then merge
        per_file = {
            (obj_id, entry.capture_ts): entry.bbox
            for obj_id, entries in tracked_objects.items()
            for entry in entries
        }
        lookup_dict.update(per_file)  # extend + overwrite on key collisions

    # make it return None on missing keys
    lookup = defaultdict(lambda: None, lookup_dict)

    # --- outputs for frames ---
    frames_dir = os.path.join(path_store, "frames")
    shards_dir = os.path.join(path_store, "frames_shards")
    if write_mode in ("files", "both"):
        os.makedirs(frames_dir, exist_ok=True)
    shard_writer = None
    if write_mode in ("shards", "both"):
        shard_writer = ShardWriter(
            out_dir=shards_dir,
            prefix=shard_prefix,
            max_shard_size_bytes=max_shard_size_bytes,
            max_samples_per_shard=max_samples_per_shard,
            flush_every=flush_every,
        )

    # async file writer pool
    executor = ThreadPoolExecutor(max_workers=workers) if write_mode in ("files", "both") else None
    pending, written_since_flush = [], 0

    def submit_write(basename_no_ext: str, ext: str, data_bytes: bytes):
        fname = os.path.join(frames_dir, f"{basename_no_ext}.{ext}")
        return executor.submit(atomic_write_bytes, fname, data_bytes)

    # --- frame index csv (covers both modes) ---
    index_path = os.path.join(path_store, "index.csv")
    write_header = not os.path.exists(index_path)
    index_file = open(index_path, "a", newline="", buffering=1)
    index_writer = csv.writer(index_file)
    if write_header:
        index_writer.writerow(["filename", "timestamp_utc_ms", "frame_index", "shard"])

    # --- detections output ---
    det_list = []   # only used if detections_out_format == "json"
    det_path_json = os.path.join(path_store, "object_detections.json")
    det_fp = None

    try:
        for path_sae_dump in paths_sae_dumps:
            with open(path_sae_dump, "r") as input_file:
                messages = saedump.message_splitter(input_file)

                # Validate meta/header
                start_message = next(messages)
                saedump.DumpMeta.model_validate_json(start_message)

                for i, message in tqdm(enumerate(messages), desc="collecting frames"):
                    if max_frames_per_dump is not None and i >= max_frames_per_dump:
                        break

                    event = saedump.Event.model_validate_json(message)
                    proto_bytes = pybase64.standard_b64decode(event.data_b64)

                    proto = SaeMessage()
                    proto.ParseFromString(proto_bytes)

                    # --- frame extraction ---
                    frame_data = proto.frame.frame_data_jpeg
                    ts_ms = int(getattr(proto.frame, "timestamp_utc_ms", 0) or 0)
                    base_name = f"frame_{ts_ms:013d}_{i:06d}" if ts_ms > 0 else f"frame_no-ts_{i:06d}"
                    shard_name = ""
                    key = base_name  # key inside shard
                    frame_filename = f"{base_name}.jpg"

                    # write: files
                    if write_mode in ("files", "both"):
                        fut = submit_write(base_name, "jpg", frame_data)
                        pending.append(fut)
                        written_since_flush += 1

                    # write: shards (sync)
                    if write_mode in ("shards", "both"):
                        shard_writer.add(key, frame_data)
                        shard_name = os.path.basename(shard_writer.current_path)

                    # index row for frames
                    index_writer.writerow([frame_filename, ts_ms, i, shard_name])

                    # --- detections for this frame (with bbox lookup) ---
                    detections_per_timestamp = []
                    for det in getattr(proto, "detections", []):
                        obj_id = str(det.object_id.hex())
                        bbox = lookup[(obj_id, ts_ms)]
                        if bbox is None:
                            continue
                        boundingbox = [
                            round(bbox[0][0], 4), round(bbox[0][1], 4),
                            round(bbox[1][0], 4), round(bbox[1][1], 4)
                        ]
                        detections_per_timestamp.append({
                            "class_id": det.class_id,
                            "object_id": obj_id,
                            "longitude": round(det.geo_coordinate.longitude, 5),
                            "latitude": round(det.geo_coordinate.latitude, 5),
                            "boundingbox": boundingbox,
                            "confidence": round(det.confidence, 4)
                        })

                    det_record = {
                        "timestamp": ts_ms,
                        "frame_index": i,
                        "frame_key": frame_filename,  # exact filename
                        "shard": shard_name,          # empty string if not using shards
                        "detections": detections_per_timestamp,
                    }

                    det_list.append(det_record)

                    # periodic flush (files + index + jsonl)
                    if write_mode in ("files", "both") and written_since_flush >= flush_every:
                        for f in pending:
                            f.result()
                        pending.clear()
                        written_since_flush = 0
                        index_file.flush(); os.fsync(index_file.fileno())
                        if det_fp is not None:
                            det_fp.flush()

        # drain pending file writes
        if write_mode in ("files", "both"):
            for f in pending:
                f.result()
            pending.clear()

        # finalize detections
        with open(det_path_json, "w", encoding="utf-8") as fjson:
            json.dump(det_list, fjson, indent=2, ensure_ascii=False)

        # final fsync for CSV
        index_file.flush()
        os.fsync(index_file.fileno())

    finally:
        index_file.close()
        if det_fp is not None:
            det_fp.flush()
            det_fp.close()
        if shard_writer is not None:
            shard_writer.close()
        if executor is not None:
            executor.shutdown(wait=True)

    print("Done.")


