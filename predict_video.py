import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from collections import deque

# Load Model
model = load_model('violence_detection_model002.h5')

# Initialize webcam
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('violence_1.mp4')  # Replace webcam with demo video

# Optional: Prediction history for smoothing
prediction_history = deque(maxlen=8)

# FPS calculation
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize & normalize
    frame_resized = cv2.resize(frame, (224, 224)) / 255.0
    input_tensor = np.expand_dims(frame_resized, axis=0)

    # Prediction
    pred = model.predict(input_tensor)[0][0]
    label = "Violence" if pred > 0.5 else "Non-Violence"
    confidence = round(pred * 100 if pred > 0.5 else (1 - pred) * 100, 2)

    # Smooth output
    prediction_history.append(label)
    stable_label = max(set(prediction_history), key=prediction_history.count)

    # Draw prediction
    color = (0, 0, 255) if stable_label == "Violence" else (0, 255, 0)
    cv2.putText(frame, f"{stable_label} - {confidence}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # FPS
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time))
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    # Show frame
    cv2.imshow("Violence Detection", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
