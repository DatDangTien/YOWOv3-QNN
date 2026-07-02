import os
import io
import time
import torch
import asyncio
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

import sys
# Add parent dir to path so we can import from model, utils, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.build_config import build_config
from utils.box import non_max_suppression

from qai_appbuilder import (
    QNNContext,
    PerfProfile,
    Runtime,
    LogLevel,
    ProfilingLevel,
    QNNConfig,
)

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

QNNConfig.Config(Runtime.HTP, LogLevel.WARN, ProfilingLevel.BASIC)

model_path = os.getenv("QNN_MODEL_PATH", "weights/qnn/yowov3.bin")
print(f"Loading QNN model from {model_path}")
model = QNNContext("yowov3", str(model_path))

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
    image_resized = image.resize((config['img_size'], config['img_size']))
    normalized_frame = np.array(image_resized, dtype=np.float32) / 255.0
    
    clip_np = np.stack([normalized_frame] * 16, axis=0)
    clip_np = np.transpose(clip_np, (3, 0, 1, 2))
    clip_np = np.expand_dims(clip_np, axis=0)
    
    PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)
    outputs_np = model.Inference([clip_np])[0]
    PerfProfile.RelPerfProfileGlobal()
    
    outputs = torch.tensor(outputs_np)
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
            
            image_resized = image.resize((config['img_size'], config['img_size']))
            normalized_frame = np.array(image_resized, dtype=np.float32) / 255.0
            frame_list.append(normalized_frame)
            
            if len(frame_list) > 16:
                frame_list.pop(0)
            if len(frame_list) < 16:
                await websocket.send_json({"status": "buffering", "frames": len(frame_list)})
                continue
                
            clip_np = np.stack(frame_list, axis=0)
            clip_np = np.transpose(clip_np, (3, 0, 1, 2))
            clip_np = np.expand_dims(clip_np, axis=0)
            
            PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)
            outputs_np = model.Inference([clip_np])[0]
            PerfProfile.RelPerfProfileGlobal()
            
            outputs = torch.tensor(outputs_np)
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

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

def main():
    uvicorn.run("api.main_qnn:app", host=HOST, port=PORT, reload=True)


if __name__ == "__main__":
    main()
