# YOWOv3: An Efficient and Generalized Framework for Human Action Detection and Recognition (Extended)

This repository is an extended production-ready version of [YOWOv3](https://github.com/hope1337/YOWOv3), featuring ONNX compilation pipelines, real-time streaming APIs (REST/WebSocket), Dockerized deployment config, and optimization guides for Qualcomm edge devices (QNN).

---

## 🌟 What's New?

Here is a summary of the additions and enhancements introduced in this extended version:

### 1. ONNX Optimization & Streaming Inference
*   **Export Optimization**: Modified the 3D Backbone pooling layer in [model/backbone3D/i3d.py](file:///data21tb/tiendat/YOWOv3/model/backbone3D/i3d.py) to replace dynamic `avg_pool3d` with `mean(dim=2, keepdim=True)`. This resolves shape serialization limitations in ONNX/QNN compilers.

### 2. Production REST & WebSocket APIs
*   **FastAPI Backend**: Implemented [api/main.py](file:///data21tb/tiendat/YOWOv3/api/main.py) which exposes:
    *   `POST /predict`: Standard HTTP endpoint for single static frame analysis (frames replicated to satisfy temporal volume).
    *   `WS /ws`: High-performance WebSocket endpoint streaming binary video frames. Utilizes frame skipping, buffer depletion logic, and dynamic client/server sync.
*   **WebSocket Client**: Created [test_ws_client.py](file:///data21tb/tiendat/YOWOv3/test_ws_client.py) to stream local/network video files to the WebSocket server and write real-time detection overlays to an output video.

### 3. Containerized GPU Deployment
*   Added a [Dockerfile](file:///data21tb/tiendat/YOWOv3/Dockerfile) and [docker-compose.yml](file:///data21tb/tiendat/YOWOv3/docker-compose.yml) leveraging PyTorch CUDA runtime, with GPU allocation configurations ready for production deployment.

---

## 🚀 Qualcomm Edge Device Target (QNN Integration)

YOWOv3 can be converted and executed on Qualcomm Snapdragon edge platforms (Hexagon DSP/NPU) using the **Qualcomm Neural Network (QNN)** SDK. 

### Conversion to QNN DLC Format
Our change to `i3d.py` (replacing dynamic `avg_pool3d` with `mean`) enables smooth compilation through the QNN tools. Use the following steps to convert the optimized model:

1.  **Generate ONNX model**:
    ```bash
    python main.py --mode onnx --config <config_file>
    ```
2.  **Convert ONNX to Qualcomm DLC (Deep Learning Container)**:
    ```bash
    python scripts/qnn.py --model <onnx model dir> --out-dir weights/qnn --device <Qualcomm device>
    ```
3.  **Quantize the DLC for NPU Execution**:
    Snapdragon NPUs operate most efficiently with 8-bit quantization (`INT8`). Quantize the model using representative calibration data (saving 16-frame clip tensors to binary raw file format):
    ```bash
    python scripts/qnn.py --model <onnx model dir> --out-dir weights/qnn --device <Qualcomm device> --quantize [--calib-dir <images_dir>]
    ```
    If no calib-dir provided, the calib data is genderated randomly.

### Inference on QNN
To execute the compiled model on device, you can:
*   **ONNX Runtime QNN Execution Provider**: Load the original/optimized ONNX model using the QNN Execution Provider (EP) in ONNX Runtime:
    ```python
    import onnxruntime as ort
    
    session = ort.InferenceSession(
        "yowov3.onnx",
        providers=["QnnExecutionProvider"],
        provider_options=[{
            "backend_path": "libQnnHtp.so" # Hexagon HTP Backend
        }]
    )
    ```
*   **QNN Native SDK (`qnn-net-run`)**: Run the quantized context binary directly on target Snapdragon platforms:
    ```bash
    qnn-net-run --model yowov3_quantized.dlc --backend libQnnHtp.so --input_list input_data.txt
    ```

---

## 🔧 Preparation & Setup

### Environment Setup
You can set up the environment using pip.

    Create a clean Python 3.8+ environment and run:
    ```bash
    pip install -r requirements.txt
    ```

## 🛠️ Basic Usage

Configure model hyperparameters and datasets directly in yaml config files located in [config/cf2/](file:///data21tb/tiendat/YOWOv3/config/cf2/).

Command template:
```bash
python main.py --mode [mode] --config [config_file_path]
```
Where `[mode]` can be:
*   `train`: Train the model.
*   `eval`: Evaluate performance metrics.
*   `detect`: Visualize predictions on the specified dataset.
*   `live`: Stream webcam input to the model.
*   `onnx`: Export PyTorch checkpoint to ONNX file format and run test video inference.

### Training example:
```bash
python main.py --mode train --config config/cf2/ucf_config.yaml
```

### Evaluation example:
```bash
python main.py -m eval -cf config/cf2/ava_config.yaml
```

---

## 🐳 Docker Deployment

To spin up the FastAPI streaming server inside a Docker container with NVIDIA GPU acceleration:

1.  **Build and run the container**:
    ```bash
    docker-compose up --build -d
    ```
2.  **Test the WebSocket server**:
    Run the WebSocket test client using an input video to verify inference and output rendering:
    ```bash
    python test_ws_client.py --video your_video.mp4 --url ws://localhost:8090/ws
    ```
    This outputs a processed video file named `output_test.mp4` overlaying the detected bounding boxes and action class tags.


## 👥 References
This project relies on contributions from the following repositories:
*   [YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt)
*   [Efficient-3DCNNs](https://github.com/okankop/Efficient-3DCNNs)
*   [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d)
*   [YOWO](https://github.com/wei-tim/YOWO)
*   [YOWOv2](https://github.com/yjh0410/YOWOv2)
*   [YOWOv3](https://github.com/hope1337/YOWOv3)
*   [ActivityNet Evaluation](https://github.com/activitynet/ActivityNet/tree/master/Evaluation)

---

## Citation

If you use this repository in your research, please consider citing the original YOWOv3 paper:

```latex
@misc{dang2024yowov3efficientgeneralizedframework,
      title={YOWOv3: An Efficient and Generalized Framework for Human Action Detection and Recognition}, 
      author={Duc Manh Nguyen Dang and Viet Hang Duong and Jia Ching Wang and Nhan Bui Duc},
      year={2024},
      eprint={2408.02623},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.02623}, 
}
```