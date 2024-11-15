from picamera2 import Picamera2
from libcamera import Transform
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from libcamera import controls
import time
import numpy as np
import os
from datetime import datetime

# Parameters
PIXEL_DIFF_THRESHOLD = 25  # Minimum pixel intensity difference to count as "changed"
SENSITIVITY = 500  # Number of changed pixels required to trigger motion
MOTION_BUFFER_DURATION = 1.0  # Minimum duration (in seconds) to keep "motion detected" state
frame_width, frame_height = 640, 480  # Capture resolution
motion_frame_width, motion_frame_height = 320, 240  # Resolution for motion detection
record_duration_after_motion = 10  # seconds
output_folder = "motion_videos"  # Folder to save videos
cooldown_duration = 5  # Cooldown duration in seconds

# Flip configuration
flip_horizontal = True  # Set to True to flip the image horizontally
flip_vertical = True    # Set to True to flip the image vertically

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize Picamera2
picam2 = Picamera2()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

# Configure the camera for both preview and motion detection frames
transform = Transform(hflip=flip_horizontal, vflip=flip_vertical)
video_config = picam2.create_video_configuration(
    main={"size": (frame_width, frame_height)},
    lores={"size": (motion_frame_width, motion_frame_height), "format": "YUV420"},
    display="lores",
    transform=transform
)
picam2.configure(video_config)

# Start the camera
picam2.start(show_preview=True)

motion_detected = False
recording = False
motion_end_time = None
motion_buffer_end_time = 0  # Initialize to zero to avoid NoneType issues
cooldown_end_time = 0  # Initialize cooldown period to zero
output_file = None
encoder = None
output = None

try:
    while True:
        # Capture frame for motion detection
        frame = picam2.capture_array("lores")  # Capture from the low-resolution stream

        # Check if the captured frame has color channels
        if frame.ndim == 3 and frame.shape[2] == 3:
            # Convert to grayscale by averaging the color channels
            motion_frame = np.mean(frame, axis=2).astype(np.uint8)
        else:
            # If already grayscale, use the frame as is
            motion_frame = frame

        # Initialize last_frame if it's the first loop
        if 'last_frame' not in locals():
            last_frame = motion_frame

        # Calculate frame difference and threshold it
        frame_delta = np.abs(motion_frame.astype(np.int16) - last_frame.astype(np.int16))
        changed_pixels = np.sum(frame_delta > PIXEL_DIFF_THRESHOLD)

        # Check if currently within the cooldown period
        current_time = time.time()
        if current_time < cooldown_end_time:
            # Skip motion detection during cooldown
            motion_detected = False
        else:
            # Determine if there's motion based on the count of changed pixels
            if changed_pixels > SENSITIVITY:
                if not motion_detected:
                    print("Motion detected.")
                motion_buffer_end_time = current_time + MOTION_BUFFER_DURATION
                motion_detected = True
            elif current_time > motion_buffer_end_time:
                motion_detected = False

        # Start recording if motion is detected and not currently recording
        if motion_detected and not recording:
            recording = True
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_folder, f"motion_{timestamp}.h264")
            print(f"Recording started: {output_file}")
            encoder = H264Encoder(bitrate=10000000)  # 10 Mbps bitrate
            output = FileOutput(output_file)
            picam2.start_recording(encoder, output)
            motion_end_time = current_time + record_duration_after_motion

        # Stop recording if motion has ceased for the specified time
        if recording and current_time > motion_end_time:
            print(f"Stopping recording: {output_file}")
            picam2.stop_recording()
            recording = False
            time.sleep(0.1)  # Short delay to allow the camera to stabilize

            # Reset encoder and output resources after stopping
            encoder = None
            output = None

            # Set cooldown period after stopping recording
            cooldown_end_time = current_time + cooldown_duration

            # Ensure preview continues smoothly
            picam2.start(show_preview=True)
            print("Preview resumed with cooldown")

        # Update for the next frame comparison
        last_frame = motion_frame

        # Add a delay to prevent high CPU usage
        time.sleep(0.1)

except KeyboardInterrupt:
    pass
finally:
    # Stop preview and camera
    if recording:
        picam2.stop_recording()
    picam2.stop_preview()
    picam2.close()
