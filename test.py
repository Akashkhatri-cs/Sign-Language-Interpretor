import cv2
import numpy as np
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.websockets import WebSocketDisconnect

app = FastAPI()

# Serve static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Use Jinja2 templates
templates = Jinja2Templates(directory="templates")

cap = cv2.VideoCapture(0)

async def video_stream(websocket: WebSocket):
    await websocket.accept()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to JPEG format
        _, jpeg = cv2.imencode(".jpg", frame)

        # Send the frame to the WebSocket
        await websocket.send_bytes(jpeg.tobytes())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await video_stream(websocket)
    except WebSocketDisconnect:
        await websocket.close()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("new 8.html", {"request": request})
