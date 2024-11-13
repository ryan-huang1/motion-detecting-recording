import cv2
import time
import numpy as np
import os
from picamera2 import Picamera2, Preview

# Parameters
SENSITIVITY_THRESHOLD = 300  # Lower value increases sensitivity
MOTION_BUFFER_DURATION = 1.0  # Minimum duration (in seconds) to keep "motion detected" state
frame_width, frame_height = 640, 480  # Capture resolution
motion_frame_width, motion_frame_height = 320, 240  # Resolution for motion detection
fps = 20
record_duration_after_motion = 10  # seconds
output_folder = "motion_videos"  # Folder to save videos

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

motion_detected = False
recording = False
motion_end_time = None
motion_buffer_end_time = 0  # Initialize to zero to avoid NoneType issues
output_file = None

# Initialize Picamera2
picam2 = Picamera2()

# Configure the camera
video_config = picam2.create_video_configuration(
    main={"size": (frame_width, frame_height)},
    lores={"size": (motion_frame_width, motion_frame_height)},
    display="lores"
)
picam2.configure(video_config)

# Start the camera
picam2.start()

# Create background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# Define codec and create VideoWriter object for MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Main loop
while True:
    # Capture frame
    frame = picam2.capture_array()
    if frame is None:
        print("Failed to grab frame.")
        break

    # Resize frame for motion detection
    motion_frame = cv2.resize(frame, (motion_frame_width, motion_frame_height))

    # Convert to grayscale for motion detection
    gray_motion_frame = cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction
    fgmask = fgbg.apply(gray_motion_frame)

    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Determine if there's motion based on sensitivity threshold
    detected = any(cv2.contourArea(c) > SENSITIVITY_THRESHOLD for c in contours)

    # Update motion buffer timing
    current_time = time.time()
    if detected:
        motion_buffer_end_time = current_time + MOTION_BUFFER_DURATION
        motion_detected = True
    elif current_time > motion_buffer_end_time:
        motion_detected = False

    # Start recording if motion is detected
    if motion_detected:
        if not recording:
            recording = True
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_folder, f"motion_{timestamp}.mp4")
            out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
            print(f"Recording started: {output_file}")
        motion_end_time = current_time + record_duration_after_motion

    # Stop recording if motion has ceased for the specified time
    if recording and current_time > motion_end_time:
        recording = False
        out.release()
        print(f"Recording stopped: {output_file}")

    # Write frame to video if recording
    if recording:
        out.write(frame)

    # Display preview with motion and recording indicators
    indicator_text = f"Motion: {'Yes' if motion_detected else 'No'} | Recording: {'Yes' if recording else 'No'}"
    cv2.putText(frame, indicator_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if recording else (0, 255, 0), 2)
    cv2.imshow('Camera Preview', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
picam2.stop()
if recording:
    out.release()
cv2.destroyAllWindows()
