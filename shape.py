import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, 
                      min_detection_confidence=0.75, 
                      min_tracking_confidence=0.75)
mpDraw = mp.solutions.drawing_utils

# Variables for drawing
drawing = False
points = []  # To store points while drawing
shape_detected = None  # Detected shape
button_selected = None  # Currently selected button ("draw" or "clear")
last_button_time = 0  # Timestamp of the last button click

# Button positions
button_draw_pos = (50, 20, 150, 70)  # (x1, y1, x2, y2) for the "Draw" button
button_clear_pos = (200, 20, 300, 70)  # (x1, y1, x2, y2) for the "Clear" button

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access webcam.")
        break

    # Flip the frame and preprocess
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    # Draw buttons
    cv2.rectangle(frame, (button_draw_pos[0], button_draw_pos[1]), 
                  (button_draw_pos[2], button_draw_pos[3]), (255, 0, 0), -1)
    cv2.putText(frame, "Draw", (button_draw_pos[0] + 10, button_draw_pos[1] + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.rectangle(frame, (button_clear_pos[0], button_clear_pos[1]), 
                  (button_clear_pos[2], button_clear_pos[3]), (0, 0, 255), -1)
    cv2.putText(frame, "Clear", (button_clear_pos[0] + 10, button_clear_pos[1] + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Check for hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

            # Get fingertip position (index finger)
            fore_finger = hand_landmarks.landmark[8]
            fx, fy = int(fore_finger.x * frame.shape[1]), int(fore_finger.y * frame.shape[0])

            # Get thumb position
            thumb = hand_landmarks.landmark[4]
            tx, ty = int(thumb.x * frame.shape[1]), int(thumb.y * frame.shape[0])

            # Check if thumb and index are close (pinch gesture)
            pinch = np.hypot(tx - fx, ty - fy) < 30

            # Check if the index finger is clicking a button
            current_time = time.time()
            if fy < 100:  # Within the button area
                if button_draw_pos[0] <= fx <= button_draw_pos[2] and current_time - last_button_time > 2:
                    button_selected = "draw"
                    last_button_time = current_time  # Update button press timestamp
                elif button_clear_pos[0] <= fx <= button_clear_pos[2] and current_time - last_button_time > 2:
                    button_selected = "clear"
                    points = []  # Clear points
                    shape_detected = None
                    last_button_time = current_time  # Update button press timestamp

            # Draw a small circle at the fingertip
            cv2.circle(frame, (fx, fy), 8, (0, 255, 0), -1)

            # Start/stop drawing based on the selected mode and pinch gesture
            if button_selected == "draw" and not pinch:
                points.append((fx, fy))
            elif pinch:
                points.append(None)  # Add a break in the drawing when pinched

    # Draw the points on the canvas
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        cv2.line(frame, points[i - 1], points[i], (0, 0, 255), 2)

    # Detect circle after drawing is complete
    if not drawing and button_selected == "draw" and len(points) > 10:
        contour = np.array([p for p in points if p is not None], dtype=np.int32)
        if len(contour) > 0:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            distances = [np.linalg.norm(np.array([cx, cy]) - np.array(p)) for p in contour]
            if np.var(distances) < 500:  # Circle detection threshold
                shape_detected = "Circle"
            else:
                shape_detected = None

    # Display the detected shape
    if shape_detected:
        cv2.putText(frame, f"Detected: {shape_detected}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Shape Drawing Tool", frame)

    # Exit the program when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
