import os
import io
import time
import torch
import asyncio
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.responses import JSONResponse

import sys
# Add parent dir to path so we can import from model, utils, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.build_config import build_config
from model.TSN.YOWOv3 import build_yowov3
from utils.box import non_max_suppression

import torchvision.transforms.functional as FT

class live_transform():
    def __init__(self, img_size):
        self.img_size = img_size

    def to_tensor(self, image):
        return FT.to_tensor(image)
    
    def normalize(self, clip):
        mean  = torch.FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std   = torch.FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        clip -= mean
        clip /= std
        return clip
    
    def __call__(self, img):
        img = img.resize([self.img_size, self.img_size])
        img = self.to_tensor(img)
        img = self.normalize(img)
        return img

def group_detections(outputs, mapping, scale_x=1.0, scale_y=1.0):
    grouped = []
    if outputs is not None and len(outputs):
        for *box, conf, cls in outputs.cpu().numpy():
            b = [
                float(box[0]) * scale_x,
                float(box[1]) * scale_y,
                float(box[2]) * scale_x,
                float(box[3]) * scale_y
            ]
            action = {
                "class_id": int(cls),
                "class_name": mapping.get(int(cls), "Unknown"),
                "confidence": float(conf)
            }
            
            matched = False
            for gd in grouped:
                gb = gd["box"]
                iou_x1, iou_y1 = max(b[0], gb[0]), max(b[1], gb[1])
                iou_x2, iou_y2 = min(b[2], gb[2]), min(b[3], gb[3])
                inter_area = max(0, iou_x2 - iou_x1) * max(0, iou_y2 - iou_y1)
                b_area = (b[2] - b[0]) * (b[3] - b[1])
                gb_area = (gb[2] - gb[0]) * (gb[3] - gb[1])
                iou = inter_area / (b_area + gb_area - inter_area + 1e-6)
                
                if iou > 0.95:
                    gd["actions"].append(action)
                    matched = True
                    break
                    
            if not matched:
                grouped.append({
                    "box": b,
                    "actions": [action]
                })
    return grouped

app = FastAPI()

config_path = os.getenv("CONFIG_PATH", "weights/ava/SE/config.yaml")
config = build_config(config_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}")
model = build_yowov3(config)
model.to(device)
model.eval()

transform = live_transform(config['img_size'])
mapping = config['idx2name']

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    start_time = time.time()
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    orig_w, orig_h = image.size
    scale_x = orig_w / config['img_size']
    scale_y = orig_h / config['img_size']
    
    # Since YOWOv3 expects a clip of 16 frames, we replicate the image 16 times for a static prediction
    frame_tensor = transform(image)
    clip = torch.stack([frame_tensor] * 16, 0).permute(1, 0, 2, 3).contiguous().unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(clip)
        outputs = non_max_suppression(outputs, conf_threshold=0.5, iou_threshold=0.5)[0]
    
    detections = group_detections(outputs, mapping, scale_x, scale_y)
            
    latency = time.time() - start_time
    return JSONResponse(content={"latency_s": latency, "detections": detections})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    frame_list = []
    
    try:
        while True:
            # Buffer skipping logic: read all available messages and keep only the latest
            data = None
            try:
                # Read from websocket non-blocking (drain buffer)
                while True:
                    data = await asyncio.wait_for(websocket.receive_bytes(), timeout=0.001)
            except asyncio.TimeoutError:
                pass
            except Exception:
                break
            
            if data is None:
                # If no data is available in buffer, wait for the next frame blockingly
                try:
                    data = await websocket.receive_bytes()
                except Exception:
                    break
                
            start_time = time.time()
            
            image = Image.open(io.BytesIO(data)).convert("RGB")
            orig_w, orig_h = image.size
            scale_x = orig_w / config['img_size']
            scale_y = orig_h / config['img_size']
            frame_tensor = transform(image)
            frame_list.append(frame_tensor)
            
            if len(frame_list) > 16:
                frame_list.pop(0)
            if len(frame_list) < 16:
                await websocket.send_json({"status": "buffering", "frames": len(frame_list)})
                continue
                
            clip = torch.stack(frame_list, 0).permute(1, 0, 2, 3).contiguous().unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(clip)
                outputs = non_max_suppression(outputs, conf_threshold=0.5, iou_threshold=0.5)[0]
            
            detections = group_detections(outputs, mapping, scale_x, scale_y)
            
            print(f"Detect: {len(detections)} boxes")

            latency = time.time() - start_time
            await websocket.send_json({
                "status": "success",
                "latency_s": latency, 
                "detections": detections
            })
            
    except WebSocketDisconnect:
        print("Client disconnected")
