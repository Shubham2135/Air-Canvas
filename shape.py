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
button_selected = None  # Currently selected button ("pen" or "eraser")
last_button_time = 0  # Timestamp of the last button click
hand_present = False  # To track if the hand is in the frame
shape_detected = False  # Flag to indicate if shapes have been detected
shapes_info = []  # To store detected shapes

# Button positions
button_pen_pos = (50, 20, 150, 70)  # (x1, y1, x2, y2) for the "Pen" button
button_eraser_pos = (200, 20, 300, 70)  # (x1, y1, x2, y2) for the "Eraser" button

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
    cv2.rectangle(frame, (button_pen_pos[0], button_pen_pos[1]),
                  (button_pen_pos[2], button_pen_pos[3]), (255, 0, 0), -1)
    cv2.putText(frame, "Pen", (button_pen_pos[0] + 10, button_pen_pos[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.rectangle(frame, (button_eraser_pos[0], button_eraser_pos[1]),
                  (button_eraser_pos[2], button_eraser_pos[3]), (0, 0, 255), -1)
    cv2.putText(frame, "Eraser", (button_eraser_pos[0] + 10, button_eraser_pos[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Check for hand landmarks
    if result.multi_hand_landmarks:
        shapes_info = []  # Reset detected shapes when the hand is present
        shape_detected = False
        hand_present = True  # Hand is detected in the frame
        for hand_landmarks in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

            # Get fingertip positions (index and middle fingers)
            index_finger = hand_landmarks.landmark[8]
            middle_finger = hand_landmarks.landmark[12]
            fx, fy = int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])
            mx, my = int(middle_finger.x * frame.shape[1]), int(middle_finger.y * frame.shape[0])

            # Calculate distance between index and middle fingers
            distance = np.hypot(mx - fx, my - fy)

            # Check finger states
            index_up = index_finger.y < hand_landmarks.landmark[6].y  # Index finger is up
            middle_up = middle_finger.y < hand_landmarks.landmark[10].y  # Middle finger is up

            # Check if the index finger is clicking a button
            current_time = time.time()
            if fy < 100:  # Within the button area
                if button_pen_pos[0] <= fx <= button_pen_pos[2] and current_time - last_button_time > 2:
                    button_selected = "pen"
                    last_button_time = current_time
                elif button_eraser_pos[0] <= fx <= button_eraser_pos[2] and current_time - last_button_time > 2:
                    button_selected = "eraser"
                    last_button_time = current_time

            # Draw a small circle at the fingertip
            cv2.circle(frame, (fx, fy), 8, (0, 255, 0), -1)

            # Handle drawing/erasing based on the selected mode
            if button_selected == "pen" and index_up and not middle_up:
                if not drawing:  # Add a break if restarting drawing
                    points.append(None)
                points.append((fx, fy))
                drawing = True
            elif button_selected == "pen" and middle_up:
                drawing = False
            elif button_selected == "eraser" and index_up and middle_up:
                for i, point in enumerate(points):
                    if point is not None:  # Skip None values
                        px, py = point
                        if np.hypot(px - fx, py - fy) < distance:
                            points[i] = None  # Erase point

                # Display eraser mode
                cv2.putText(frame, "Eraser Mode", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        if hand_present:  # If hand was present but now gone
            points.append(None)
            hand_present = False

    # If hand is not in the frame and shapes have not been detected yet
    if not hand_present and not shape_detected and points:
        # Create a blank canvas to draw points
        canvas = np.zeros_like(frame)
        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            cv2.line(canvas, points[i - 1], points[i], (255, 255, 255), 2)

        # Convert canvas to grayscale and find contours
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Identify the shape based on the number of vertices
            x, y, w, h = cv2.boundingRect(approx)
            if len(approx) == 3:
                shape_name = "Triangle"
                color = (0, 255, 255)
            elif len(approx) == 4:
                aspect_ratio = float(w) / h
                shape_name = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
                color = (255, 0, 255)
            elif len(approx) > 4:
                shape_name = "Circle"
                color = (0, 255, 0)
            else:
                shape_name = "Unknown"
                color = (0, 0, 255)

            # Save shape information for persistent display
            shapes_info.append((shape_name, x, y, w, h, color))

        shape_detected = True  # Prevent detecting shapes repeatedly

    # Draw the detected shapes
    for shape_name, x, y, w, h, color in shapes_info:
        cv2.putText(frame, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if shape_name == "Circle":
            center = (x + w // 2, y + h // 2)
            radius = w // 2
            cv2.circle(frame, center, radius, color, 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Draw the points on the canvas
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        cv2.line(frame, points[i - 1], points[i], (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Drawing Tool", frame)

    # Exit the program when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
