import sys
import cv2
import numpy as np
import mediapipe as mp
import time
from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtCore import Qt, QTimer


class AirCanvasApp(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize MediaPipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mpDraw = mp.solutions.drawing_utils

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)

        # UI Setup
        self.setWindowTitle("Air Canvas")
        self.setGeometry(100, 100, 1024, 768)  # Adjusted window size
        self.layout = QVBoxLayout()

        # Canvas (video feed)
        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        # Button Layout
        self.button_layout = QHBoxLayout()
        self.pen_button = QPushButton("Pen")
        self.eraser_button = QPushButton("Eraser")

        self.button_layout.addWidget(self.pen_button)
        self.button_layout.addWidget(self.eraser_button)
        self.layout.addLayout(self.button_layout)

        # Set selected drawing mode to "pen"
        self.button_selected = "pen"

        # Variables for drawing
        self.points = []
        self.drawing = False
        self.last_button_time = 0

        self.setLayout(self.layout)
        self.show()

        # Timer for updating webcam feed and processing frame
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Flip the frame and preprocess
        frame = cv2.flip(frame, 1)

        # Add Pen and Eraser Buttons inside the camera
        cv2.rectangle(frame, (50, 20), (150, 70), (255, 0, 0), -1)  # Pen Button
        cv2.putText(frame, "Pen", (60, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (200, 20), (300, 70), (0, 0, 255), -1)  # Eraser Button
        cv2.putText(frame, "Eraser", (210, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Convert the frame to RGB for PySide2 compatibility
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(framergb)

        # Check for hand landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame, hand_landmarks, self.mpHands.HAND_CONNECTIONS)

                # Get fingertip positions (index, middle, and thumb fingers)
                index_finger = hand_landmarks.landmark[8]
                middle_finger = hand_landmarks.landmark[12]
                thumb_finger = hand_landmarks.landmark[4]
                fx, fy = int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])
                mx, my = int(middle_finger.x * frame.shape[1]), int(middle_finger.y * frame.shape[0])
                tx, ty = int(thumb_finger.x * frame.shape[1]), int(thumb_finger.y * frame.shape[0])

                # Calculate distance between index and thumb fingers (for Pen selection)
                thumb_index_distance = np.hypot(tx - fx, ty - fy)

                # Handle drawing/erasing based on the selected mode
                if thumb_index_distance < 50:  # Close enough for Pen selection
                    self.button_selected = "pen"
                elif index_finger.y < hand_landmarks.landmark[6].y and middle_finger.y < hand_landmarks.landmark[10].y:
                    self.button_selected = "eraser"

                # Handle Pen and Eraser Mode
                if self.button_selected == "pen" and index_finger.y < hand_landmarks.landmark[6].y and middle_finger.y > hand_landmarks.landmark[10].y:
                    if not self.drawing:  # Add a break if restarting drawing
                        self.points.append(None)
                    self.points.append((fx, fy))
                    self.drawing = True
                elif self.button_selected == "pen" and middle_finger.y < hand_landmarks.landmark[10].y:
                    self.drawing = False
                elif self.button_selected == "eraser" and index_finger.y < hand_landmarks.landmark[6].y and middle_finger.y < hand_landmarks.landmark[10].y:
                    for i, point in enumerate(self.points):
                        if point is not None:  # Skip None values
                            px, py = point
                            if np.hypot(px - fx, py - fy) < 30:
                                self.points[i] = None  # Erase point

        # Draw the points on the canvas
        for i in range(1, len(self.points)):
            if self.points[i - 1] is None or self.points[i] is None:
                continue
            cv2.line(frame, self.points[i - 1], self.points[i], (0, 0, 255), 2)

        # Convert the frame to QImage and update the QLabel
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        """Release the webcam and close the application gracefully."""
        self.cap.release()
        event.accept()


# Run the app
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AirCanvasApp()
    sys.exit(app.exec_())
