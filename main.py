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

# Canvas and drawing variables
base_canvas = None
current_strokes = []
canvas_visible = False
drawing = False
button_selected = "pen"
last_button_time = 0
shapes_info = []

# Button positions
buttons = {
    "pen": (50, 20, 150, 70, (255, 0, 0)),
    "eraser": (200, 20, 300, 70, (0, 0, 255)),
    "shape": (350, 20, 450, 70, (0, 255, 0)),
    "save": (500, 20, 600, 70, (0, 255, 255)),
    "open": (650, 20, 750, 70, (255, 0, 255))
}

# Initialize webcam
cap = cv2.VideoCapture(0)

def detect_shapes(canvas):
    shapes = []
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

def handle_canvas_operations():
    global base_canvas, canvas_visible
    try:
        loaded = cv2.imread("drawing.png")
        if loaded is not None:
            base_canvas = loaded
        else:
            base_canvas = np.ones_like(frame) * 255
    except:
        base_canvas = np.ones_like(frame) * 255
    canvas_visible = True
    cv2.namedWindow("Canvas", cv2.WINDOW_NORMAL)

def draw_buttons(frame):
    for name, (x1, y1, x2, y2, color) in buttons.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.putText(frame, name.capitalize(), (x1 + 10, y1 + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access webcam.")
        break

    # Initialize canvas
    if base_canvas is None:
        base_canvas = np.ones_like(frame) * 255

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    # Draw buttons
    draw_buttons(frame)

    # Mode display
    mode_text = f"Current Mode: {button_selected.capitalize()}"
    color = buttons[button_selected][4] if button_selected in buttons else (255,255,255)
    cv2.putText(frame, mode_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
            
            # Get finger positions
            index_finger = hand_landmarks.landmark[8]
            middle_finger = hand_landmarks.landmark[12]
            fx, fy = int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])
            mx, my = int(middle_finger.x * frame.shape[1]), int(middle_finger.y * frame.shape[0])

            # Button interactions
            current_time = time.time()
            if fy < 100 and current_time - last_button_time > 1:
                for name, (x1, y1, x2, y2, _) in buttons.items():
                    if x1 <= fx <= x2 and y1 <= fy <= y2:
                        if name == "save":
                            cv2.imwrite("drawing.png", base_canvas)
                            cv2.destroyWindow("Canvas")
                            canvas_visible = False
                        elif name == "open":
                            handle_canvas_operations()
                        else:
                            button_selected = name
                        last_button_time = current_time

            # Drawing operations
            if button_selected == "pen":
                index_up = index_finger.y < hand_landmarks.landmark[6].y
                middle_up = middle_finger.y < hand_landmarks.landmark[10].y
                
                if index_up and not middle_up:
                    current_strokes.append((fx, fy))
                    drawing = True
                elif drawing:
                    # Commit stroke to base canvas
                    for i in range(1, len(current_strokes)):
                        cv2.line(base_canvas, current_strokes[i-1], current_strokes[i], (0,0,255), 2)
                    current_strokes = []
                    drawing = False
                    
            elif button_selected == "eraser":
                if index_up and middle_up:
                    distance = np.hypot(mx-fx, my-fy)
                    cv2.circle(base_canvas, (fx, fy), int(distance*1.5), (255,255,255), -1)

            elif button_selected == "shape":
                shapes_info = detect_shapes(base_canvas)

    # Update canvas display
    if canvas_visible:
        display_canvas = base_canvas.copy()
        # Draw current strokes
        for i in range(1, len(current_strokes)):
            cv2.line(display_canvas, current_strokes[i-1], current_strokes[i], (0,0,255), 2)
        # Draw detected shapes
        for shape in shapes_info:
            name, x, y, w, h, color = shape
            cv2.putText(display_canvas, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(display_canvas, (x, y), (x+w, y+h), color, 2)
        cv2.imshow("Canvas", display_canvas)

    cv2.imshow("Drawing Tool", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()