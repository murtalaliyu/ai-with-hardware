import cv2
import mediapipe as mp
import math
from mediapipe.tasks.python.components.containers import landmark
import numpy as np
import time
import serial

# Calculate the euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# Calculate the angle formed by a-b-c, where b is the vertex
def calculate_angle(a, b, c):
    AB = (a.x - b.x, a.y - b.y)     # Vector from b to a
    BC = (c.x - b.x, c.y - b.y)     # Vector from b to c

    # Calculate the dot product of the vectors
    dot_product = AB[0] * BC[0] + AB[1] * BC[1]

    # Calculate the lengths of the vectors
    magnitude_AB = math.sqrt(AB[0] ** 2 + AB[1] ** 2)
    magnitude_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)

    # Calculate cos(theta)
    cos_angle = dot_product / (magnitude_AB * magnitude_BC)

    # Use arccos to calculate the angle in degrees
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))
    return angle

def is_finger_straight(landmarks, finger_tip_index, finger_mcp_index, wrist_index=0):
    # Determine if a finger is straight (based on distance)
    wrist = landmarks[wrist_index]
    finger_tip = landmarks[finger_tip_index]
    finger_mcp = landmarks[finger_mcp_index]

    tip_to_wrist_distance = calculate_distance(finger_tip, wrist)
    mcp_to_wrist_distance = calculate_distance(finger_mcp, wrist)

    return tip_to_wrist_distance > mcp_to_wrist_distance

def is_finger_bent(landmarks, mcp_index, pip_index, dip_index):
    # Determine if a finger is bent (based on angle)
    angle = calculate_angle(landmarks[mcp_index], landmarks[pip_index], landmarks[dip_index])

    # Consider the finger bent if angle is less than 160 degrees 
    return angle < 160

def count_fingers(hand_landmarks):
    landmarks = hand_landmarks.landmark
    finger_states = [
        # Thumb
        not is_finger_bent(landmarks, 1, 2, 3) and is_finger_straight(landmarks, 4, 1),
        # Index
        not is_finger_bent(landmarks, 5, 6, 7) and is_finger_straight(landmarks, 8, 5),
        # Middle
        not is_finger_bent(landmarks, 9, 10, 11) and is_finger_straight(landmarks, 12, 9),
        # Ring
        not is_finger_bent(landmarks, 13, 14, 15) and is_finger_straight(landmarks, 16, 13),
        # Pinky
        not is_finger_bent(landmarks, 17, 18, 19) and is_finger_straight(landmarks, 20, 17)
    ]

    return sum(finger_states), finger_states

def create_info_panel(finger_info, image_width, panel_height=50):
    # Shape must be a tuple; 3 channels so it stacks with the BGR camera frame
    panel = np.zeros((panel_height, image_width, 3), dtype=np.uint8)

    # Create a string to store the number of hands and the number of fingers for each hand
    info_text = f"Hands: {len(finger_info)} | "
    for i, count in enumerate(finger_info):
        info_text += f"Hand {i+1}: {count} fingers | "

    # If no hands are detected, display appropriate information
    if not finger_info:
        info_text = "No hands detected"

    # Draw text on the panel
    cv2.putText(panel, info_text.strip(), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

    return panel

# Create a holder for caputres video
cap = cv2.VideoCapture(0)

# Detect the key points on hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Adjust the port to match our setup
port = "COM4"
# This rate is decided by the hardware
baudrate = 115200

# Serial connection to talk to STEPico
serial_connection = serial.Serial(port, baudrate)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        finger_info = []
        total_fingers = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                finger_count, _ = count_fingers(hand_landmarks)
                finger_info.append(finger_count)
                total_fingers += finger_count

        # Print information in the terminal
        print(f"Hands: {len(finger_info)} | Finger counts: {finger_info} | Total fingers: {total_fingers}")

        if len(finger_info) != 0:
            if len(finger_info) > 0:
                serial_connection.write(('1' + str(finger_info[0])).encode())

            if len(finger_info) > 1:
                serial_connection.write(('2' + str(finger_info[1])).encode())
            else:
                serial_connection.write(('R').encode())
            
        else:
            serial_connection.write(('L').encode())

        serial_connection.write(('V').encode())

        # Get image width
        image_width = image.shape[1]

        # Create information panel
        info_panel = create_info_panel(finger_info, image_width)

        # Flip the main image
        image = cv2.flip(image, 1)

        # Vertically stack the main image and information panel
        display_image = np.vstack((image, info_panel))

        # Display the result in a window
        cv2.imshow('MediaPipe Hands', display_image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
serial_connection.close()
cv2.destroyAllWindows()

