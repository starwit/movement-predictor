import subprocess
from PIL import Image
from io import BytesIO
import time
import logging
from collections import defaultdict
from datetime import datetime, timezone
import cv2
import numpy as np
import os
import json
import zipfile
from zlib import crc32


log = logging.getLogger(__name__)


def storeVideo(frames, path, tracks, anomalies, log_level):
    log.setLevel(log_level)
    seconds_passed = (frames[-1].timestamp_utc_ms - frames[0].timestamp_utc_ms)/1_000
    framerate = len(frames) / seconds_passed
    ffmpeg_command = [
            'ffmpeg', '-f', 'image2pipe', '-r', str(framerate),
            '-vcodec', 'mjpeg', '-i', '-', '-c:v', 'libx264', '-crf', '25',
            '-analyzeduration', '2147483647', '-probesize', '2147483647', '-y', f'{path}/out.mp4'
        ]
    
    try:
        process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        tracks_unusual_movement = filter_relevant_tracks(tracks, anomalies)
        start_time_millis = time.time() * 1000
        try:
            for frame in frames:
                annotated_frame = draw_trajectories_in_frame(frame, tracks_unusual_movement)
                process.stdin.write(annotated_frame)
                process.stdin.flush()
            log.debug("Average time per frame: " + str((time.time() * 1000 - start_time_millis) / len(frames)))
        except Exception as ex:
            log.error("Could not create video from frames")
            log.debug(str(ex))
            log.debug(process.stderr.read().decode())
            

        process.stdin.close()

        process.wait()

        if process.returncode == 0:
            log.info("Video conversion completed successfully.")
        else:
            log.error("Video conversion failed.")
            log.debug(process.stderr.read().decode())
    except (IOError, subprocess.SubprocessError) as e:
        print(e)

    
def filter_relevant_tracks(detections, anomalies):
    tracks = defaultdict(list)
    anomaly_ids = list({anomaly[-1] for anomaly in anomalies})
    tracks_of_interest = [detection for id, detection in detections.items() if id in anomaly_ids]
    flattened_tracks_of_interest = [item for track_sublist in tracks_of_interest for item in track_sublist]

    frame_ts = {track.capture_ts for track in flattened_tracks_of_interest}
    for fr_ts in frame_ts:
        tracks[fr_ts] = []

    for track in flattened_tracks_of_interest:
        tracks[track.capture_ts].append(track)

    return tracks


def draw_trajectories_in_frame(frame, tracks):
    center_points = []

    scaling_factor_x = frame.shape.width
    scaling_factor_y = frame.shape.height

    for tracks_per_frame in tracks.values():
        for track in tracks_per_frame:
            if track.capture_ts == datetime.fromtimestamp(frame.timestamp_utc_ms/1_000, tz=timezone.utc):
                center_points.append(track.center)

    frame_data = frame.frame_data_jpeg
    np_arr = np.frombuffer(frame_data, np.uint8)
    frame_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

    for point in center_points:
        x, y = point
        x = int(x * scaling_factor_x)
        y = int(y * scaling_factor_y)
        cv2.circle(frame_img, (x, y), 20, (0, 0, 255), -1)

    _, encoded_img = cv2.imencode('.jpeg', frame_img)
    return encoded_img.tobytes()


def store_frames(frames, path, tracks, anomalies, log_level):
    log.setLevel(log_level)
    seconds_passed = (frames[-1].timestamp_utc_ms - frames[0].timestamp_utc_ms)/1_000
    framerate = len(frames) / seconds_passed

    tracks_unusual_movement = filter_relevant_tracks(tracks, anomalies)

    scaling_factors = [frames[0].shape.width, frames[0].shape.height]
    metadata_json = convert_metadata_to_json(framerate, scaling_factors, tracks_unusual_movement)

    write_json_to_file(metadata_json, os.path.join(path, "metadata.json"))

    create_and_add_frames_to_zip(frames, os.path.join(path, "frames.zip"))


def convert_metadata_to_json(framerate, scaling_factors, tracks_unusual_movement):
    metadata_json = {
        "framerate": framerate,
        "scalingFactors": scaling_factors,
        "tracksUnusualMovement": {}
    }

    for frame_ts, tracks in tracks_unusual_movement.items():
        metadata_json["tracksUnusualMovement"][frame_ts.isoformat()] = [track.to_json() for track in tracks]

    return metadata_json


def write_json_to_file(json_data, file_path):
    try:
        with open(file_path, 'w') as file_writer:
            json.dump(json_data, file_writer, indent=2)
    except IOError as e:
        print(f"Could not store JSON file {file_path}")
        print(e)


def create_and_add_frames_to_zip(frames, zip_file_path):
    try:
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_STORED) as zip_output_stream:
            for frame in frames:
                data = frame.frame_data_jpeg 
                entry_name = f"frameTs_{frame.timestamp_utc_ms}.jpg"
                
                crc = crc32(data) & 0xffffffff

                zip_info = zipfile.ZipInfo(entry_name)
                zip_info.compress_type = zipfile.ZIP_STORED
                zip_info.file_size = len(data)
                zip_info.CRC = crc

                zip_output_stream.writestr(zip_info, data)
    except IOError as e:
        print(f"Could not store frames in zip-file {zip_file_path}")
        print(e)