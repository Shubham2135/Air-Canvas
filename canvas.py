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
points = []
button_selected = None
last_button_time = 0
hand_present = False
shapes_info = []

# Button positions
button_pen_pos = (50, 20, 150, 70)    # Pen button
button_eraser_pos = (200, 20, 300, 70) # Eraser button
button_shape_pos = (350, 20, 450, 70)  # Shape detection button

# Initialize webcam
cap = cv2.VideoCapture(0)

def detect_shapes(points, frame):
    shapes = []
    canvas = np.zeros_like(frame)
    
    # Draw all points on canvas
    for i in range(1, len(points)):
        if points[i-1] and points[i]:
            cv2.line(canvas, points[i-1], points[i], (255,255,255), 2)
    
    # Process canvas for shape detection
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
            aspect_ratio = float(w)/h
            shape_name = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
            color = (255, 0, 255)
        elif len(approx) > 6:
            area = cv2.contourArea(contour)
            circle_area = np.pi * (w/2) * (h/2)
            if abs(area - circle_area) < 1000:
                shape_name = "Circle"
                color = (0, 255, 0)
            else:
                continue
        else:
            continue
        
        shapes.append((shape_name, x, y, w, h, color))
    
    return shapes

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access webcam.")
        break

    # Flip and process frame
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    # Draw buttons
    cv2.rectangle(frame, button_pen_pos[:2], button_pen_pos[2:], (255,0,0), -1)
    cv2.putText(frame, "Pen", (button_pen_pos[0]+10, button_pen_pos[1]+40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    cv2.rectangle(frame, button_eraser_pos[:2], button_eraser_pos[2:], (0,0,255), -1)
    cv2.putText(frame, "Eraser", (button_eraser_pos[0]+10, button_eraser_pos[1]+40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    cv2.rectangle(frame, button_shape_pos[:2], button_shape_pos[2:], (0,255,0), -1)
    cv2.putText(frame, "Shape", (button_shape_pos[0]+10, button_shape_pos[1]+40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Mode display
    mode_text = "Current Mode: "
    if button_selected == "pen":
        mode_text += "Pen Mode"
        color = (255, 0, 0)
    elif button_selected == "eraser":
        mode_text += "Eraser Mode"
        color = (0, 0, 255)
    elif button_selected == "shape":
        mode_text += "Shape Detection Mode"
        color = (0, 255, 0)
    else:
        mode_text += "None"
        color = (255, 255, 255)
    
    cv2.putText(frame, mode_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Hand detection and processing
    if result.multi_hand_landmarks:
        hand_present = True
        shapes_info = []  # Clear shapes when hand is present
        for hand_landmarks in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

            # Get finger positions
            index_finger = hand_landmarks.landmark[8]
            middle_finger = hand_landmarks.landmark[12]
            fx, fy = int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])
            mx, my = int(middle_finger.x * frame.shape[1]), int(middle_finger.y * frame.shape[0])

            # Calculate distance between fingers
            distance = np.hypot(mx - fx, my - fy)

            # Check finger states
            index_up = index_finger.y < hand_landmarks.landmark[6].y
            middle_up = middle_finger.y < hand_landmarks.landmark[10].y

            # Button selection
            current_time = time.time()
            if fy < 100:
                if button_pen_pos[0] <= fx <= button_pen_pos[2] and current_time - last_button_time > 1:
                    button_selected = "pen"
                    last_button_time = current_time
                elif button_eraser_pos[0] <= fx <= button_eraser_pos[2] and current_time - last_button_time > 1:
                    button_selected = "eraser"
                    last_button_time = current_time
                elif button_shape_pos[0] <= fx <= button_shape_pos[2] and current_time - last_button_time > 1:
                    button_selected = "shape"
                    last_button_time = current_time

            # Draw finger position marker
            cv2.circle(frame, (fx, fy), 8, (0, 255, 0), -1)

            # Handle drawing modes
            if button_selected == "pen":
                if index_up and not middle_up:
                    if not drawing:
                        points.append(None)
                    points.append((fx, fy))
                    drawing = True
                else:
                    drawing = False
                    
            elif button_selected == "eraser":
                if index_up and middle_up:
                    # Erase points within eraser radius
                    for i, point in enumerate(points):
                        if point is not None:
                            px, py = point
                            if np.hypot(px - fx, py - fy) < distance:
                                points[i] = None

    # Shape detection when in shape mode
    if button_selected == "shape":
        shapes_info = detect_shapes(points, frame)

    # Draw detected shapes
    for shape in shapes_info:
        name, x, y, w, h, color = shape
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if name == "Circle":
            cv2.circle(frame, (x+w//2, y+h//2), w//2, color, 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Draw all points
    for i in range(1, len(points)):
        if points[i-1] and points[i]:
            cv2.line(frame, points[i-1], points[i], (0,0,255), 2)

    cv2.imshow("Drawing Tool", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()