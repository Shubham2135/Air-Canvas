import cv2
import numpy as np
import mediapipe as mp
import pytesseract
import time
import os

# Path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
button_shape_pos = (350, 20, 450, 70)  # (x1, y1, x2, y2) for the "Shape Detection" button
button_text_pos = (500, 20, 600, 70)  # (x1, y1, x2, y2) for the "Text Detection" button

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

    cv2.rectangle(frame, (button_shape_pos[0], button_shape_pos[1]),
                  (button_shape_pos[2], button_shape_pos[3]), (0, 255, 0), -1)
    cv2.putText(frame, "Shape", (button_shape_pos[0] + 10, button_shape_pos[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.rectangle(frame, (button_text_pos[0], button_text_pos[1]),
                  (button_text_pos[2], button_text_pos[3]), (255, 255, 0), -1)
    cv2.putText(frame, "Text", (button_text_pos[0] + 10, button_text_pos[1] + 40),
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
            fx, fy = int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])

            # Check if the index finger is clicking a button
            current_time = time.time()
            if fy < 100:  # Within the button area
                if button_pen_pos[0] <= fx <= button_pen_pos[2] and current_time - last_button_time > 2:
                    button_selected = "pen"
                    last_button_time = current_time
                elif button_eraser_pos[0] <= fx <= button_eraser_pos[2] and current_time - last_button_time > 2:
                    button_selected = "eraser"
                    last_button_time = current_time
                elif button_shape_pos[0] <= fx <= button_shape_pos[2] and current_time - last_button_time > 2:
                    os.system("python shape.py")
                    last_button_time = current_time
                elif button_text_pos[0] <= fx <= button_text_pos[2] and current_time - last_button_time > 2:
                    os.system("python text.py")
                    last_button_time = current_time

            # Draw a small circle at the fingertip
            cv2.circle(frame, (fx, fy), 8, (0, 255, 0), -1)

            # Handle drawing/erasing based on the selected mode
            if button_selected == "pen":
                points.append((fx, fy))
                drawing = True
            elif button_selected == "eraser":
                points.clear()
                drawing = False

    else:
        if hand_present:  # If hand was present but now gone
            points.append(None)
            hand_present = False

    # If hand is not in the frame, detect possible text-like areas
    if not hand_present and points:
        # Create a blank canvas to draw points
        canvas = np.zeros_like(frame)
        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            cv2.line(canvas, points[i - 1], points[i], (255, 255, 255), 2)

        # Convert canvas to grayscale and apply thresholding
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Use pytesseract to extract text from the image
        text = pytesseract.image_to_string(thresh, config='--psm 6')

        # Display the detected text
        if text.strip():
            cv2.putText(frame, f"Detected Text: {text}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
