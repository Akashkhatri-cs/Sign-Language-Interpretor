# Bring in lighwright dependencied
from fastapi import FastAPI, File, Request, UploadFile, HTTPException, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from keras.models import load_model
import numpy as np
import cv2
from starlette.websockets import WebSocketDisconnect
import mediapipe as mp

app = FastAPI()

model = load_model('sign_asl_cnn_30_epochs.h5')

# @app.get('/')
# async def root():
#     return {"Hello":"World"}

class_labels = {i: str(i) if i < 10 else chr(65 + i - 10) for i in range(36)}
def preprocess_image(image):
    # Resize the image to match the input size of your model
    image = cv2.resize(image, (200, 200))
    
    # Normalize the pixel values to the range [0, 1]
    image = image / 255.0
    
    # Reshape the image to match the input shape of your model
    image = image.reshape(1, 200, 200, 3)
    
    return image

def predict_letter(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Make predictions using the model
    predictions = model.predict(processed_image)
    
    # Get the predicted class (assuming it's the class with the highest probability)
    predicted_class = np.argmax(predictions, axis=1)[0]
    # Map the predicted class to the corresponding letter or label
    
    sign_letter = class_labels[predicted_class] #chr(65 + predicted_class) if predicted_class < 26 else str(predicted_class - 26)
    
    return sign_letter

@app.post("/predict")
async def predict_sign_language(file: UploadFile = File(...)):
    try:
        # Read the image from the uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Make the prediction
        predicted_letter = predict_letter(image)
        
        return JSONResponse(content={"predicted_letter": predicted_letter}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


# OpenCV video capture
cap = cv2.VideoCapture(0)

# Create a FastAPI templates and static folder for HTML response
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Define the main route for capturing video and making predictions
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# WebSocket route for live video stream
# WebSocket route for live video stream
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # message = await websocket.receive_text()
            # if message == "stop":
            #     break
            # else:
            #     cap =cv2.VideoCapture(0)
            #     await websocket.accept()
            ret, frame = cap.read()
            if not ret:
                break

            # Detect hands in the current frame
            frame = detect_hands(frame)

            # Convert the frame to JPEG format
            _, jpeg = cv2.imencode(".jpg", frame)

            # Send the frame to the WebSocket
            await websocket.send_bytes(jpeg.tobytes())
    except WebSocketDisconnect:
        cap.release()
        await websocket.close()


def detect_hands(image):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    margin=15
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

            # # Check if the ROI is empty
            if roi.size == 0:
                continue

            # Resize the ROI to match your model's input shape
            roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
            lower_yellow = np.array([93, 72, 51])
            upper_yellow = np.array([224, 194, 183])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            roi = cv2.bitwise_and(roi,roi, mask= mask)
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
            # Draw landmarks on the image
            # mp.solutions.drawing_utils.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

    return image