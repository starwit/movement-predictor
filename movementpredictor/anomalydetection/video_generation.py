import subprocess
import time
import logging
import cv2
import numpy as np


log = logging.getLogger(__name__)


def store_video(frame_data, path):
    seconds_passed = (frame_data[-1][0].timestamp_utc_ms - frame_data[0][0].timestamp_utc_ms)/1_000
    framerate = len(frame_data) / seconds_passed
    ffmpeg_command = [
            'ffmpeg', '-f', 'image2pipe', '-r', str(framerate),
            '-vcodec', 'mjpeg', '-i', '-', '-c:v', 'libx264', '-crf', '25',
            '-analyzeduration', '2147483647', '-probesize', '2147483647', '-y', f'{path}/out.mp4'
        ]
    
    try:
        process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        start_time_millis = time.time() * 1000
        try:
            for frame, bbox in frame_data:
                annotated_frame = draw_bbox_in_frame(frame, bbox)
                process.stdin.write(annotated_frame)
                process.stdin.flush()
            log.debug("Average time per frame: " + str((time.time() * 1000 - start_time_millis) / len(frame_data)))
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
        log(f"Could not complete ffmped command, error: {e}")


def draw_bbox_in_frame(frame, bbox):
    frame_data = frame.frame_data_jpeg
    np_arr = np.frombuffer(frame_data, np.uint8)
    frame_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

    if bbox is not None:
        center_point = [(bbox.min_x + bbox.max_x)/2, (bbox.min_y + bbox.max_y)/2]

        scaling_factor_x = frame_img.shape[1]
        scaling_factor_y = frame_img.shape[0]

        x = int(center_point[0] * scaling_factor_x)
        y = int(center_point[1] * scaling_factor_y)
        cv2.circle(frame_img, (x, y), 5, (0, 0, 255), -1)

    _, encoded_img = cv2.imencode('.jpeg', frame_img)
    return encoded_img.tobytes()