import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Canvas and tool variables
canvas = None
canvas_visible = False
image_path = "drawing.png"
base_canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White canvas
current_strokes = []
button_selected = "pen"
last_button_time = 0

# Button positions and parameters
buttons = {
    "pen": (50, 20, 150, 70, (255, 0, 0)),
    "eraser": (200, 20, 300, 70, (0, 0, 255)),
    "save": (350, 20, 450, 70, (0, 255, 0)),
    "open": (500, 20, 600, 70, (0, 255, 255))
}

# Initialize webcam
cap = cv2.VideoCapture(0)

def handle_canvas_operations():
    global base_canvas, canvas_visible
    try:
        base_canvas = cv2.imread(image_path)
        if base_canvas is None:
            base_canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
    except Exception as e:
        print(f"Error loading canvas: {e}")
        base_canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
    canvas_visible = True
    cv2.namedWindow("Canvas", cv2.WINDOW_AUTOSIZE)

def draw_buttons(frame):
    for name, (x1, y1, x2, y2, color) in buttons.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.putText(frame, name.capitalize(), (x1 + 10, y1 + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access webcam.")
        break

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(framergb)

    draw_buttons(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get finger landmarks
            index = hand_landmarks.landmark[8]
            middle = hand_landmarks.landmark[12]
            fx, fy = int(index.x * frame.shape[1]), int(index.y * frame.shape[0])
            mx, my = int(middle.x * frame.shape[1]), int(middle.y * frame.shape[0])
            
            # Calculate finger distance
            finger_distance = np.hypot(fx - mx, fy - my)
            fingers_together = finger_distance < 30

            # Handle button clicks
            current_time = time.time()
            if fy < 100 and current_time - last_button_time > 1:
                for name, (x1, y1, x2, y2, _) in buttons.items():
                    if x1 <= fx <= x2 and y1 <= fy <= y2:
                        if name == "open":
                            handle_canvas_operations()
                        elif name == "save":
                            cv2.imwrite(image_path, base_canvas)
                            cv2.destroyWindow("Canvas")
                            canvas_visible = False
                        else:
                            button_selected = name
                        last_button_time = current_time

            # Drawing operations
            if button_selected == "pen":
                if not fingers_together:
                    current_strokes.append((fx, fy))
                elif current_strokes:
                    # Commit current stroke to base canvas
                    for i in range(1, len(current_strokes)):
                        cv2.line(base_canvas, current_strokes[i-1], current_strokes[i], (0, 0, 255), 2)
                    current_strokes.clear()
            elif button_selected == "eraser":
                eraser_size = int(finger_distance * 1.5)
                cv2.circle(base_canvas, (fx, fy), max(5, eraser_size), (255, 255, 255), -1)

    # Update canvas display
    if canvas_visible:
        display_canvas = base_canvas.copy()
        # Draw current pen strokes
        for i in range(1, len(current_strokes)):
            cv2.line(display_canvas, current_strokes[i-1], current_strokes[i], (0, 0, 255), 2)
        cv2.imshow("Canvas", display_canvas)

    cv2.imshow("Drawing Tool", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()