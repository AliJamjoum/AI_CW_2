import cv2
import mediapipe as mp

# Load image
image_path = "../example_image.jpg"  # adjust if needed
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialise MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# Process image
results = hands.process(image_rgb)

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

# Show result
cv2.imshow("MediaPipe Hand Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
