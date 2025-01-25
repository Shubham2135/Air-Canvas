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
button_selected = None  # Currently selected button ("draw" or "clear")
last_button_time = 0  # Timestamp of the last button click
last_drawing_time = None  # Time of the last drawing action
shape_locked = False  # Lock the shape after detection
detected_shape = None  # Name of the detected shape
shape_contour = None  # Contour of the detected shape

# Button positions
button_draw_pos = (50, 20, 150, 70)  # (x1, y1, x2, y2) for the "Draw" button
button_clear_pos = (200, 20, 300, 70)  # (x1, y1, x2, y2) for the "Clear" button

# Initialize webcam
cap = cv2.VideoCapture(0)

def fit_shape(points):
    """Convert rough drawings to proper shapes."""
    filtered_points = [p for p in points if p is not None]  # Remove None values
    if len(filtered_points) < 5:  # Too few points to detect a shape
        return None, None

    # Create a contour from the points
    contour = np.array(filtered_points, dtype=np.int32)

    # Approximate the contour to detect shapes
    approx = cv2.approxPolyDP(contour, epsilon=5, closed=True)

    if len(approx) == 3:
        return "Triangle", approx
    elif len(approx) == 4:
        # Check if the shape is a rectangle or square
        rect = cv2.boundingRect(approx)
        aspect_ratio = rect[2] / rect[3]  # width/height
        if 0.9 <= aspect_ratio <= 1.1:
            return "Square", approx
        else:
            return "Rectangle", approx
    else:
        # Fit a circle
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        distances = [np.linalg.norm(np.array([cx, cy]) - np.array(p)) for p in filtered_points]
        if np.var(distances) < 1000:  # Circle detection threshold
            return "Circle", None

    return None, None

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
                    last_button_time = current_time
                elif button_clear_pos[0] <= fx <= button_clear_pos[2] and current_time - last_button_time > 2:
                    button_selected = "clear"
                    points = []
                    detected_shape = None
                    shape_locked = False
                    last_button_time = current_time

            # Draw a small circle at the fingertip
            cv2.circle(frame, (fx, fy), 8, (0, 255, 0), -1)

            # Start/stop drawing based on the selected mode and pinch gesture
            if button_selected == "draw" and not pinch:
                points.append((fx, fy))
                drawing = True
                last_drawing_time = time.time()  # Update last drawing time
                shape_locked = False
            elif pinch and drawing:
                points.append(None)  # Add a break in the drawing when pinched
                drawing = False

    # Detect and convert rough drawing to proper shape after 3 seconds of inactivity
    if not drawing and points and last_drawing_time and time.time() - last_drawing_time > 3 and not shape_locked:
        detected_shape, shape_contour = fit_shape(points)
        shape_locked = True

    # Draw the detected shape
    if shape_locked and detected_shape:
        if detected_shape == "Circle":
            filtered_points = [p for p in points if p is not None]
            (cx, cy), radius = cv2.minEnclosingCircle(np.array(filtered_points, dtype=np.int32))
            cv2.circle(frame, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
        elif shape_contour is not None:
            cv2.polylines(frame, [shape_contour], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, f"Detected: {detected_shape}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw the points on the canvas
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        cv2.line(frame, points[i - 1], points[i], (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Shape Drawing Tool", frame)

    # Exit the program when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
