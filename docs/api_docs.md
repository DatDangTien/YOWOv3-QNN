# YOWOv3 WebSocket API Documentation

This document describes the WebSocket streaming API for the YOWOv3 action detection model.

## Endpoint Overview
- **URL**: `ws://<host>:<port>/ws` (e.g., `ws://localhost:8090/ws`)
- **Protocol**: WebSocket
- **Purpose**: Real-time action detection from a continuous stream of video frames.

## 1. Request Format

### WebSocket (`/ws`)
Clients should send frames individually as **raw binary messages** over the WebSocket connection. 

- **Data Type**: **Raw Binary Bytes** (Do **NOT** use Base64 encoding, JSON, or form-data). You simply send the byte array of the compressed image file directly over the socket.
- **Image Format**: Any common compressed image format supported by Python's `Pillow` library (e.g., **.jpg / .jpeg**, **.png**, **.webp**). **JPEG** is highly recommended for optimal network compression and speed during streaming.

## 2. Response Format

The server responds to *every* frame it processes with a JSON text message.

Because the YOWOv3 model requires a sequence of **16 frames** to make a prediction, the first 15 frames will result in a "buffering" response. From the 16th frame onwards, it will return the actual detection results.

### A. Buffering Response
Returned when the server is still collecting the initial 16 frames.

```json
{
  "status": "buffering",
  "frames": 5
}
```
| Field | Type | Description |
|---|---|---|
| `status` | `string` | Always `"buffering"`. |
| `frames` | `integer` | The current number of frames collected (from 1 to 15). |


### B. Success Response (Detection)
Returned for every frame once the 16-frame buffer is full.

```json
{
  "status": "success",
  "latency_s": 0.045,
  "detections": [
    {
      "box": [120.5, 45.2, 300.1, 400.8],
      "actions": [
        {
          "class_id": 11,
          "class_name": "stand",
          "confidence": 0.98
        },
        {
          "class_id": 12,
          "class_name": "talk to (e.g., self, a person, a group)",
          "confidence": 0.85
        }
      ]
    }
  ]
}
```
| Field | Type | Description |
|---|---|---|
| `status` | `string` | Always `"success"`. |
| `latency_s` | `float` | The end-to-end processing latency for this specific frame, measured in seconds (from frame reception to prediction completion). |
| `detections` | `array` | A list of detected action bounding boxes. Empty array `[]` if nothing is detected. |

#### Detection Object Details
| Field | Type | Description |
|---|---|---|
| `box` | `array[float]` | The bounding box coordinates `[x1, y1, x2, y2]`. |
| `actions` | `array[object]` | A list of actions detected for this bounding box. |

#### Action Object Details
| Field | Type | Description |
|---|---|---|
| `confidence` | `float` | The confidence score of the detection (0.0 to 1.0). |
| `class_id` | `integer` | The internal ID of the detected action. |
| `class_name` | `string` | The human-readable name of the action (e.g., "walk", "stand", "sit"). |
