import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Deques to store the points for drawing
bpoints = [deque(maxlen=1024)]  # For the color to draw with
blue_index = 0

# Kernel for dilation
kernel = np.ones((5, 5), np.uint8)

# Default color for drawing (Black)
colors = [(0, 0, 0)]  # Black color for drawing

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, 
                      min_detection_confidence=0.75, 
                      min_tracking_confidence=0.75)  # Increased confidence thresholds
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

ret = True
while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read frame from webcam.")
        break

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)

    # Preprocess the frame (adjust brightness and contrast)
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)  # Increase contrast and brightness slightly

    # Apply Gaussian Blur to reduce noise
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert the frame to RGB for MediaPipe
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # Post-process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * frame.shape[1])  # Adjust to frame dimensions
                lmy = int(lm.y * frame.shape[0])

                landmarks.append([lmx, lmy])

            # Draw landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        # Get finger coordinates for interaction
        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])

        # Draw a circle around the finger
        cv2.circle(frame, center, 8, (0, 255, 0), -1)

        # Check if the thumb is near the center of the forefinger
        if abs(thumb[1] - center[1]) < 30:
            bpoints.append(deque(maxlen=512))  # Start a new line of points for drawing
            blue_index += 1

        # If the hand is near the color selection box, clear the canvas
        if center[1] <= 65:
            if 40 <= center[0] <= 140:  # Clear Button
                bpoints = [deque(maxlen=512)]  # Reset points list
                blue_index = 0

    else:
        bpoints.append(deque(maxlen=512))  # Start new deque if no hand detected
        blue_index += 1

    # Draw the points on the camera frame
    for i in range(len(bpoints)):
        for j in range(len(bpoints[i])):
            for k in range(1, len(bpoints[i][j])):
                if bpoints[i][j][k - 1] is None or bpoints[i][j][k] is None:
                    continue
                cv2.line(frame, bpoints[i][j][k - 1], bpoints[i][j][k], colors[0], 2)

    # Show the frame with drawing and hand landmarks
    cv2.imshow("Camera Feed", frame)

    # Exit the program when the user presses 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
