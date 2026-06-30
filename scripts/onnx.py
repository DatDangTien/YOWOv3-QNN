from model.TSN.YOWOv3 import build_yowov3
from model.TSN.YOWOv3Normalized import YOWOv3Normalized
from cus_datasets.build_dataset import build_dataset
from utils.box import non_max_suppression
import onnxruntime

import torch
from utils.box import draw_bounding_box
import cv2
import numpy as np

def export2onnx(config):
    model   = build_yowov3(config) 
    model.eval()

    # Wrap the model to handle normalization in-graph
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalized_model = YOWOv3Normalized(model, mean, std)
    normalized_model.eval()

    # Export using float32 dummy input of shape [B, C, T, H, W] in range [0.0, 1.0]
    dummy_input = torch.randn(1, 3, 16, 224, 224, dtype=torch.float32)

    torch.onnx.export(normalized_model,
                    dummy_input,
                    "yowov3.onnx",
                    verbose=False,
                    input_names=['clip'],
                    output_names=['image'],
                    export_params=True)
    
    
    mapping = config['idx2name']
    onnx_model_path = "yowov3.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    video_path = "test_video_h264.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0

    # Output video writer
    output_path = "test_video_h264_out.mp4"
    img_size = config['img_size']
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (img_size, img_size))

    frame_queue = []
    frame_index = 0

    print("Starting video inference...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to model input size for consistency
        resized_frame = cv2.resize(frame, (img_size, img_size))

        # Preprocess frame for the clip (BGR -> RGB, then scale to [0, 1])
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        normalized_frame = rgb_frame.astype(np.float32) / 255.0

        # Maintain 16-frame sliding window
        if len(frame_queue) == 0:
            for _ in range(16):
                frame_queue.append(normalized_frame)
        else:
            frame_queue.append(normalized_frame)
            frame_queue.pop(0)

        # Stack along temporal axis -> [16, H, W, C]
        clip_np = np.stack(frame_queue, axis=0)
        # Permute to [C, T, H, W] -> [3, 16, H, W]
        clip_np = np.transpose(clip_np, (3, 0, 1, 2))
        # Add batch dimension -> [1, 3, 16, H, W] (the model expects float32 input)
        clip_np = np.expand_dims(clip_np, axis=0).astype(np.float32)

        # Inference
        input_data = {ort_session.get_inputs()[0].name: clip_np}
        outputs = torch.tensor(ort_session.run(None, input_data)[0])
        
        # Postprocessing: NMS
        det_outputs = non_max_suppression(outputs, conf_threshold=0.3, iou_threshold=0.5)[0]

        # Draw visualized bboxes on current frame
        draw_bounding_box(resized_frame, det_outputs[:, :4], det_outputs[:, 5], det_outputs[:, 4], mapping)

        # Write frame to output video
        out.write(resized_frame)

        frame_index += 1
        if frame_index % 50 == 0:
            print(f"Processed {frame_index} frames...")

    cap.release()
    out.release()
    print(f"Inference complete! Output saved to {output_path}")