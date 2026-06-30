import cv2
import asyncio
import websockets
import argparse
import json

async def stream_video(video_path, ws_url, frame_limit=None):
    # Open the video file or stream
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source at {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_test.mp4', fourcc, fps, (width, height))
    print("Saving output to output_test.mp4")

    print(f"Connecting to WebSocket at {ws_url}...")
    try:
        async with websockets.connect(ws_url) as websocket:
            print("Connected successfully. Streaming video...")
            
            frame_count = 0
            while cap.isOpened():
                if frame_limit is not None and frame_count >= frame_limit:
                    print(f"Reached frame limit of {frame_limit}. Stopping stream.")
                    break
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream.")
                    break
                
                # Encode the frame as JPEG for transmission
                success, buffer = cv2.imencode('.jpg', frame)
                if not success:
                    continue
                    
                # Send the binary JPEG data
                await websocket.send(buffer.tobytes())
                frame_count += 1
                
                # Wait for the JSON response from the server
                response = await websocket.recv()
                
                
                result_data = json.loads(response)
                print(f"--- Frame {frame_count} | Status: {result_data.get('status')} ---")
                
                # Draw boxes if we have success and detections
                if result_data.get("status") == "success":
                    for det in result_data.get("detections", []):
                        x1, y1, x2, y2 = map(int, det["box"])
                        print(f"Debug Box: {x1}, {y1}, {x2}, {y2}")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        y_offset = max(15, y1 - 10)
                        for action in det.get("actions", []):
                            label = f"{action['class_name']} ({action['confidence']:.2f})"
                            cv2.putText(frame, label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            y_offset -= 15
                
                out.write(frame)
                await asyncio.sleep(0.01)

    except ConnectionRefusedError:
        print(f"Connection refused. Is the server running at {ws_url}?")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOWOv3 WebSocket Client Tester")
    parser.add_argument("--video", type=str, required=True, help="Path or URL to the MP4 video file")
    parser.add_argument("--url", type=str, default="ws://localhost:8090/ws", help="WebSocket URL of the API (default: ws://localhost:8000/ws)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of frames to process")
    
    args = parser.parse_args()
    
    # Run the async stream
    asyncio.run(stream_video(args.video, args.url, args.limit))
