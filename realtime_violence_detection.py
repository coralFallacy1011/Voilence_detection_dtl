import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('violence_detection_model.h5')

# Set the target image size (should match the input shape of your model)
IMG_SIZE = (128, 128)  # Updated to match model's expected input size

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and preprocess the frame
    resized_frame = cv2.resize(frame, IMG_SIZE)
    normalized_frame = resized_frame / 255.0  # normalize if required
    input_frame = np.expand_dims(normalized_frame, axis=0)  # (1, height, width, 3)

    # Make prediction
    prediction = model.predict(input_frame)[0][0]  # assuming binary classification

    # Label and color based on prediction
    label = "Violence" if prediction > 0.5 else "Non-Violence"
    color = (0, 0, 255) if label == "Violence" else (0, 255, 0)

    # Draw label on the frame
    cv2.putText(frame, f"{label}: {prediction:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the frame
    cv2.imshow("Violence Detection", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
