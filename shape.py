import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# Variables for drawing
drawing = False
points = []
button_selected = None
last_button_time = 0
hand_present = False
shape_detected = False
shapes_info = []

# Button positions
button_pen_pos = (50, 20, 150, 70)
button_eraser_pos = (200, 20, 300, 70)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access webcam.")
        break

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    # Draw buttons
    cv2.rectangle(frame, (50, 20), (150, 70), (255, 0, 0), -1)
    cv2.putText(frame, "Pen", (60, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.rectangle(frame, (200, 20), (300, 70), (0, 0, 255), -1)
    cv2.putText(frame, "Eraser", (210, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if result.multi_hand_landmarks:
        shapes_info = []
        shape_detected = False
        hand_present = True
        for hand_landmarks in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

            index_finger = hand_landmarks.landmark[8]
            middle_finger = hand_landmarks.landmark[12]
            fx, fy = int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])
            mx, my = int(middle_finger.x * frame.shape[1]), int(middle_finger.y * frame.shape[0])

            distance = np.hypot(mx - fx, my - fy)

            index_up = index_finger.y < hand_landmarks.landmark[6].y
            middle_up = middle_finger.y < hand_landmarks.landmark[10].y
            current_time = time.time()

            if fy < 100:
                if button_pen_pos[0] <= fx <= button_pen_pos[2] and current_time - last_button_time > 2:
                    button_selected = "pen"
                    last_button_time = current_time
                elif button_eraser_pos[0] <= fx <= button_eraser_pos[2] and current_time - last_button_time > 2:
                    button_selected = "eraser"
                    last_button_time = current_time

            cv2.circle(frame, (fx, fy), 8, (0, 255, 0), -1)

            if button_selected == "pen" and index_up and not middle_up:
                if not drawing:
                    points.append(None)
                points.append((fx, fy))
                drawing = True
            elif button_selected == "pen" and middle_up:
                drawing = False
            elif button_selected == "eraser" and index_up and middle_up:
                points = [pt if np.hypot(pt[0] - fx, pt[1] - fy) >= distance else None for pt in points]
    else:
        if hand_present:
            points.append(None)
            hand_present = False

    if not hand_present and not shape_detected and points:
        canvas = np.zeros_like(frame)
        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            cv2.line(canvas, points[i - 1], points[i], (255, 255, 255), 2)

        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)

            if len(approx) == 3:
                shape_name = "Triangle"
                color = (0, 255, 255)
            elif len(approx) == 4:
                aspect_ratio = float(w) / h
                shape_name = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
                color = (255, 0, 255)
            elif len(approx) > 6 and abs(cv2.contourArea(contour) - (np.pi * (w / 2) * (h / 2))) < 1000:
                shape_name = "Circle"
                color = (0, 255, 0)
            else:
                shape_name = "Unknown"
                color = (0, 0, 255)

            shapes_info.append((shape_name, x, y, w, h, color))

        shape_detected = True

    for shape_name, x, y, w, h, color in shapes_info:
        cv2.putText(frame, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if shape_name == "Circle":
            center = (x + w // 2, y + h // 2)
            radius = w // 2
            cv2.circle(frame, center, radius, color, 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        cv2.line(frame, points[i - 1], points[i], (0, 0, 255), 2)

    cv2.imshow("Drawing Tool", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()