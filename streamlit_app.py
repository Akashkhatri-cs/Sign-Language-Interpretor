import streamlit as st
import numpy as np
from keras.models import load_model
import cv2
from io import BytesIO
import mediapipe as mp

# Load the model
model = load_model('sign_asl_cnn_30_epochs.h5')
class_labels = {i: str(i) if i < 10 else chr(65 + i - 10) for i in range(36)}

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (200, 200))
    image = image / 255.0
    image = image.reshape(1, 200, 200, 3)
    return image

# Function to predict the sign language letter
def predict_letter(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    sign_letter = class_labels[predicted_class]
    return sign_letter

# Function to detect hands in the image
def detect_hands(image):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    margin = 15

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and get the hand landmarks
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Get bounding box coordinates of the hand
            landmarks_xy = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                            for landmark in landmarks.landmark]

            # Define the bounding box for the hand
            x_min = max(0, min(landmarks_xy, key=lambda x: x[0])[0] - margin)
            y_min = max(0, min(landmarks_xy, key=lambda x: x[1])[1] - margin)
            x_max = min(image.shape[1], max(landmarks_xy, key=lambda x: x[0])[0] + margin)
            y_max = min(image.shape[0], max(landmarks_xy, key=lambda x: x[1])[1] + margin)

            # Extract the hand region
            roi = image[y_min:y_max, x_min:x_max]

            # Check if the ROI is empty
            if roi.size == 0:
                continue

            # Resize the ROI to match your model's input shape
            roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            lower_yellow = np.array([93, 72, 51])
            upper_yellow = np.array([224, 194, 183])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            roi = cv2.bitwise_and(roi, roi, mask=mask)
            roi = roi.reshape(1, 200, 200, 3)  # Ensure it matches your model's input shape

            # Make predictions using your classifier
            predictions = model.predict(roi)
            predicted_class = int(np.argmax(predictions, axis=1)[0])
            result = class_labels[predicted_class]

            # Draw result on the image
            cv2.putText(image, str(result), (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Draw bounding box on the image
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    return image

# Streamlit app
st.title('Sign Language Recognition')

# Sidebar with radio button for Upload/Webcam
selected_option = st.sidebar.radio("Select Option", ["Upload", "Webcam"], index=0)

if selected_option == "Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        if st.button('Predict'):
            contents = uploaded_file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Make the prediction
            predicted_letter = predict_letter(image)

            # Display the predicted letter
            st.write('Predicted Letter:', predicted_letter)

elif selected_option == "Webcam":
    # Placeholder for webcam frame
    webcam_frame = st.empty()

    # Placeholder for predicted letter in webcam mode
    predicted_letter_webcam = st.empty()

    # Placeholder for webcam capture status
    webcam_capture_status = st.empty()

    # Placeholder for webcam stop button
    webcam_stop_button = st.empty()

    # Placeholder for webcam status
    webcam_status = st.empty()

    # Placeholder for webcam button
    webcam_button = st.button("Start Webcam")

    if webcam_button:
        webcam_status.text("Webcam is on.")
        webcam_stop_button = st.button("Stop Webcam")

        # OpenCV video capture
        cap = cv2.VideoCapture(0)

        while True:
            # Read the frame from the webcam
            ret, frame = cap.read()

            # Display the frame in Streamlit
            webcam_frame.image(frame, channels="BGR")

            # Detect hands in the current frame
            frame = detect_hands(frame)

            # Convert the frame to JPEG format
            _, jpeg = cv2.imencode(".jpg", frame)

            # Display the predicted letter
            predicted_letter = predict_letter(frame)
            predicted_letter_webcam.text(f"Predicted Letter: {predicted_letter}")

            # Check if the "Stop Webcam" button is clicked
            if webcam_stop_button:
                webcam_status.text("Webcam is off.")
                break

        # Release the webcam when done
        cap.release()
