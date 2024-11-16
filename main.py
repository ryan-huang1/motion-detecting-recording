from picamera2 import Picamera2
from libcamera import Transform
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from libcamera import controls
import time
import numpy as np
import os
from datetime import datetime
import sys

# Check if running from systemd
RUNNING_FROM_SYSTEMD = bool(os.getenv('INVOCATION_ID'))  # This env var is present when running from systemd

# Parameters
PIXEL_DIFF_THRESHOLD = 25  # Minimum pixel intensity difference to count as "changed"
SENSITIVITY = 50  # Number of changed pixels required to trigger motion
MOTION_BUFFER_DURATION = 1.0  # Duration to keep "motion detected" state after motion ceases
HIGH_RES_WIDTH, HIGH_RES_HEIGHT = 1920, 1080  # High-resolution for saving
MOTION_FRAME_WIDTH, MOTION_FRAME_HEIGHT = 320, 240  # Low resolution for motion detection
RECORD_DURATION_AFTER_MOTION = 10  # Seconds to keep recording after motion stops
OUTPUT_FOLDER = "motion_videos"  # Folder to save videos
COOLDOWN_DURATION = 5  # Cooldown duration in seconds

# Flip configuration
FLIP_HORIZONTAL = True  # Set to True to flip the image horizontally
FLIP_VERTICAL = True    # Set to True to flip the image vertically

def log(message):
    """Print only if not running from systemd"""
    if not RUNNING_FROM_SYSTEMD:
        print(message)

# Ensure output folder exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Initialize Picamera2
picam2 = Picamera2()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

# Configure the camera for high-resolution main and low-resolution lores streams
transform = Transform(hflip=FLIP_HORIZONTAL, vflip=FLIP_VERTICAL)
video_config = picam2.create_video_configuration(
    main={"size": (HIGH_RES_WIDTH, HIGH_RES_HEIGHT)},  # High resolution for saved video
    lores={"size": (MOTION_FRAME_WIDTH, MOTION_FRAME_HEIGHT), "format": "YUV420"},  # Low-res for motion detection
    display="lores",
    transform=transform
)
picam2.configure(video_config)

# Start the camera with preview only if not running from systemd
picam2.start(show_preview=not RUNNING_FROM_SYSTEMD)

# Initialize state variables
motion_detected = False
recording = False
motion_end_time = 0  # Time when to stop recording
motion_buffer_end_time = 0  # Time until which motion is considered active
cooldown_end_time = 0  # Time when cooldown ends
output_file = None
encoder = None
output = None

try:
    while True:
        current_time = time.time()

        # Capture frame for motion detection
        frame = picam2.capture_array("lores")  # Capture from the low-resolution stream

        # Convert to grayscale
        if frame.ndim == 3 and frame.shape[2] == 3:
            motion_frame = np.mean(frame, axis=2).astype(np.uint8)
        else:
            motion_frame = frame

        # Initialize last_frame if it's the first loop
        if 'last_frame' not in locals():
            last_frame = motion_frame
            time.sleep(0.1)
            # Initial status print
            log(f"Motion: Not detected | Recording: {'Yes' if recording else 'No'}")
            continue  # Skip processing on the first frame

        # Calculate frame difference and threshold it
        frame_delta = np.abs(motion_frame.astype(np.int16) - last_frame.astype(np.int16))
        changed_pixels = np.sum(frame_delta > PIXEL_DIFF_THRESHOLD)

        # Determine if there's motion based on the count of changed pixels
        if current_time < cooldown_end_time:
            # Ignore motion during cooldown
            motion_detected = False
        else:
            if changed_pixels > SENSITIVITY:
                if not motion_detected:
                    log("Motion detected.")
                motion_buffer_end_time = current_time + MOTION_BUFFER_DURATION
                motion_detected = True
            else:
                if current_time > motion_buffer_end_time:
                    motion_detected = False

        # Start recording if motion is detected and not currently recording
        if motion_detected and not recording:
            recording = True
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(OUTPUT_FOLDER, f"motion_{timestamp}.h264")
            log(f"Recording started: {output_file}")
            encoder = H264Encoder(bitrate=10000000)  # 10 Mbps bitrate
            output = FileOutput(output_file)
            picam2.start_recording(encoder, output)
            motion_end_time = current_time + RECORD_DURATION_AFTER_MOTION

        # If recording is ongoing
        if recording:
            if motion_detected:
                # Extend recording time
                motion_end_time = current_time + RECORD_DURATION_AFTER_MOTION
            else:
                # Stop recording if motion has ceased and recording time has elapsed
                if current_time > motion_end_time:
                    log(f"Stopping recording: {output_file}")
                    picam2.stop_recording()
                    recording = False
                    encoder = None
                    output = None
                    cooldown_end_time = current_time + COOLDOWN_DURATION
                    log("Entered cooldown period.")

                    # Reconfigure and restart the camera
                    picam2.stop()
                    picam2.configure(video_config)
                    picam2.start(show_preview=not RUNNING_FROM_SYSTEMD)
                    log("Camera reconfigured.")

        # Update for the next frame comparison
        last_frame = motion_frame

        # Print current status
        motion_status = "Detected" if motion_detected else "Not detected"
        recording_status = "Yes" if recording else "No"
        log(f"Motion: {motion_status} | Recording: {recording_status}")

        # Add a short delay to prevent high CPU usage
        time.sleep(0.1)

except KeyboardInterrupt:
    log("Interrupted by user.")

finally:
    # Stop recording if it's still ongoing
    if recording:
        picam2.stop_recording()
        log(f"Recording stopped: {output_file}")
    # Close camera
    picam2.close()
    log("Camera closed.")