import cv2
import numpy as np
import mediapipe as mp
import time
import math
import os

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1,
                      min_detection_confidence=0.75,
                      min_tracking_confidence=0.75)
mpDraw = mp.solutions.drawing_utils

# Constants
THRESHOLD = 30  # Distance threshold for fingers together
BUTTON_DEBOUNCE = 1  # Seconds between button presses
MIN_AREA = 150       # Reduced minimum contour area
CIRCULARITY_THRESHOLD = 0.5  # More lenient circle threshold
ASPECT_RATIO_TOLERANCE = 0.3  # Increased tolerance

# Drawing state
current_stroke = []    # Points in current stroke
button_selected = None
last_button_time = 0
canvas_open = False

# Initialize canvas
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Canvas initialization
canvas_image = np.ones((height, width, 3), dtype=np.uint8) * 255
if os.path.exists('drawing.png'):
    canvas_image = cv2.imread('drawing.png')
    canvas_image = cv2.resize(canvas_image, (width, height))

# Button positions
buttons = {
    "pen": (50, 20, 150, 70),
    "eraser": (200, 20, 300, 70),
    "shape": (350, 20, 450, 70),
    "canvas": (500, 20, 600, 70)
}

def calculate_circularity(contour):
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    area = cv2.contourArea(contour)
    return (4 * math.pi * area) / (perimeter ** 2)

def improved_shape_detection(stroke):
    temp_canvas = np.zeros((height, width), dtype=np.uint8)
    
    # Draw the stroke on temporary canvas
    if len(stroke) >= 2:
        cv2.polylines(temp_canvas, [np.array(stroke)], False, 255, 2)
    
    contours, _ = cv2.findContours(temp_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        
        # Circle check
        circularity = calculate_circularity(cnt)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circle_area = math.pi * (radius ** 2)
        
        if (circularity > CIRCULARITY_THRESHOLD and 
            abs(area - circle_area) < ASPECT_RATIO_TOLERANCE * circle_area):
            return ("circle", (int(x), int(y), int(radius)))
        
        # Polygon check
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        num_vertices = len(approx)
        
        if num_vertices == 3:
            points = [tuple(point[0]) for point in approx]
            return ("triangle", points)
        
        elif num_vertices >= 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            if 0.7 <= aspect_ratio <= 1.3:
                return ("square", (x, y, w, h))
            return ("rectangle", (x, y, w, h))
    
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    # Create overlay
    overlay = frame.copy()
    
    # Always show drawing on camera feed (FIX 1: Remove canvas_open check)
    mask_white = cv2.inRange(canvas_image, (255, 255, 255), (255, 255, 255))
    mask_drawing = cv2.bitwise_not(mask_white)
    overlay_bg = cv2.bitwise_and(overlay, overlay, mask=mask_white)
    drawing_fg = cv2.bitwise_and(canvas_image, canvas_image, mask=mask_drawing)
    overlay = cv2.add(overlay_bg, drawing_fg)

    # Draw UI elements
    for name, (x1, y1, x2, y2) in buttons.items():
        display_name = name.capitalize()
        color = (255, 0, 0) if name == "pen" else \
                (0, 0, 255) if name == "eraser" else \
                (0, 255, 0) if name == "shape" else \
                (255, 255, 0)
        
        if name == "canvas":
            display_name = "Save" if canvas_open else "Canvas"
        
        if button_selected == name:
            cv2.rectangle(overlay, (x1-2, y1-2), (x2+2, y2+2), (0, 255, 0), 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.putText(overlay, display_name, (x1+10, y1+40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Mode display
    mode_text = f"Mode: {button_selected.capitalize()}" if button_selected else "Mode: None"
    cv2.putText(overlay, mode_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Hand detection
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(overlay, hand_landmarks, mpHands.HAND_CONNECTIONS)

            # Get finger positions
            index = hand_landmarks.landmark[8]
            middle = hand_landmarks.landmark[12]
            fx, fy = int(index.x * width), int(index.y * height)
            mx, my = int(middle.x * width), int(middle.y * height)

            # Calculate distance between fingers
            distance = np.hypot(mx - fx, my - fy)
            fingers_together = distance < THRESHOLD

            # Button interaction
            current_time = time.time()
            if fy < 100 and current_time - last_button_time > BUTTON_DEBOUNCE:
                for name, (x1, y1, x2, y2) in buttons.items():
                    if x1 <= fx <= x2:
                        if name == "canvas":
                            if canvas_open:
                                cv2.imwrite('drawing.png', canvas_image)
                                canvas_open = False
                                cv2.destroyWindow("Canvas")
                            else:
                                if os.path.exists('drawing.png'):
                                    canvas_image = cv2.imread('drawing.png')
                                    canvas_image = cv2.resize(canvas_image, (width, height))
                                canvas_open = True
                            last_button_time = current_time
                        else:
                            button_selected = name
                        last_button_time = current_time
                        break

            # Handle drawing modes
            if fingers_together:
                if len(current_stroke) > 0:
                    if button_selected == "pen":
                        cv2.polylines(canvas_image, [np.array(current_stroke)], False, (0,0,0), 2)
                    elif button_selected == "shape":
                        detected_shape = improved_shape_detection(current_stroke)
                        if detected_shape:
                            shape_type, data = detected_shape
                            if shape_type == "circle":
                                x, y, r = data
                                cv2.circle(canvas_image, (x, y), r, (0,0,0), 2)
                            elif shape_type == "triangle":
                                pts = np.array(data, dtype=np.int32)
                                cv2.polylines(canvas_image, [pts], True, (0,0,0), 2)
                            else:
                                x, y, w, h = data
                                cv2.rectangle(canvas_image, (x, y), (x+w, y+h), (0,0,0), 2)
                    current_stroke.clear()
            else:
                if button_selected in ["pen", "shape"]:
                    current_stroke.append((fx, fy))
                elif button_selected == "eraser":
                    radius = int(distance)
                    cv2.circle(canvas_image, (fx, fy), radius, (255,255,255), -1)
                    # Draw eraser preview (FIX 2: Add preview)
                    cv2.circle(overlay, (fx, fy), radius, (0, 0, 255), 2)

    # Draw current stroke preview
    if button_selected in ["pen", "shape"] and len(current_stroke) >= 2:
        cv2.polylines(overlay, [np.array(current_stroke)], False, (255,0,0), 2)

    # Show canvas window
    if canvas_open:
        cv2.imshow("Canvas", canvas_image)
    elif cv2.getWindowProperty("Canvas", cv2.WND_PROP_VISIBLE) > 0:
        cv2.destroyWindow("Canvas")

    cv2.imshow("Air Canvas", overlay)
    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.imwrite('drawing.png', canvas_image)
        break

cap.release()
cv2.destroyAllWindows()