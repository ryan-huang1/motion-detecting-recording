import cv2
import time
import numpy as np

# Parameters
SENSITIVITY_THRESHOLD = 150  # Lower value increases sensitivity
frame_width, frame_height = 640, 480
fps = 20
record_duration_after_motion = 10  # seconds

motion_detected = False
recording = False
motion_end_time = None
output_file = None

# Initialize Video Capture (use 0 for Pi camera)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Define codec and create VideoWriter object for MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Initialize variables for motion detection
ret, previous_frame = cap.read()
previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
previous_frame = cv2.GaussianBlur(previous_frame, (21, 21), 0)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Motion detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    frame_delta = cv2.absdiff(previous_frame, gray_frame)
    thresh = cv2.threshold(frame_delta, 20, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if motion is detected with sensitivity threshold
    motion_detected = any(cv2.contourArea(c) > SENSITIVITY_THRESHOLD for c in contours)

    if motion_detected:
        if not recording:
            recording = True
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"motion_{timestamp}.mp4"
            out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
        motion_end_time = time.time() + record_duration_after_motion

    # Stop recording if motion has ceased for the specified time
    if recording and time.time() > motion_end_time:
        recording = False
        out.release()

    # Write frame to video if recording
    if recording:
        out.write(frame)

    # Display preview with motion and recording indicators
    indicator_text = f"Motion: {'Yes' if motion_detected else 'No'} | Recording: {'Yes' if recording else 'No'}"
    cv2.putText(frame, indicator_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if recording else (0, 255, 0), 2)
    cv2.imshow('Camera Preview', frame)

    # Update previous frame
    previous_frame = gray_frame.copy()

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
if recording:
    out.release()
cv2.destroyAllWindows()
