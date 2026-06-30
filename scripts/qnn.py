import sys
import argparse
from pathlib import Path
import numpy as np
try:
    import qai_hub as hub
except ImportError:
    sys.exit("qai_hub missing — env should have qai-hub.")
import onnxruntime as Ort

MODEL_INPUT = [1, 3, 16, 224, 224]
CALIB_SIZE = 128

def make_calib_data(calib_dir: str | None):
    """Calibration entries for optional quantize. Random unless calib_dir of images given."""    
    if calib_dir:
        import cv2
        files = sorted(Path(calib_dir).glob("*"))
        samples = []
        h, w, c = MODEL_INPUT[3:], MODEL_INPUT[1]
        chunk_size = MODEL_INPUT[2]
        chunk = []
        for f in files[:CALIB_SIZE]:
            assert CALIB_SIZE % chunk_size == 0, f"CALIB_SIZE {CALIB_SIZE} must be divisible by chunk_size {chunk_size}"
            img = cv2.imread(str(f))  # BGR
            if img is None:
                continue
            img = cv2.resize(img, (w, h)).astype(np.float32) / 255.0  # host preprocess
            chunk.append(img[None, ...])
            if len(chunk) == chunk_size:
                samples.append((np.transpose(np.concatenate(chunk, axis=0), (3,0,1,2)))[None, ...]) # [1,C,D,H,W]
                chunk = []
        if not samples:
            raise RuntimeError(f"no readable images in {calib_dir}")

        return {"input": samples}
    print("[quantize] WARNING: random calibration data (low accuracy). Use --calib-dir.")
    rng = np.random.default_rng(0)
    return {"input": [rng.random(size=MODEL_INPUT, dtype=np.float32) for _ in range(CALIB_SIZE // chunk_size)]}


def quantize(hub, onnx_path: Path, calib_dir: str | None):
    """Optional INT8 quantize. Default flow skips this entirely."""
    calib = make_calib_data(calib_dir)
    print("[quantize] submitting INT8 quantize job ...")
    qjob = hub.submit_quantize_job(
        model=str(onnx_path),
        calibration_data=calib,
        name="yowov3-quant",
    )
    qmodel = qjob.get_target_model()
    if qmodel is None:
        raise RuntimeError("quantize job produced no model")
    return qmodel

def compile_qnn(hub, model_src, device):
    print(f"[compile] QNN context binary on '{device.name}' ...")
    cjob = hub.submit_compile_job(
        model=model_src,
        device=device,
        name="yowov3-qnn",
        options="--target_runtime qnn_context_binary",
    )
    target = cjob.get_target_model()
    if target is None:
        raise RuntimeError(f"compile failed: {cjob.url}")
    print(f"[compile] done: {cjob.url}")
    return cjob, target

def profile(hub, target, device):
    print(f"[profile] on '{device.name}' ...")
    pjob = hub.submit_profile_job(model=target, device=device, name="yowov3-prof")
    print(f"[profile] {pjob.url}")
    return pjob


def verify(hub, target, onnx_model, device, seed: int = 0):
    """On-device vs reference numeric parity on one sample tensor."""
    rng = np.random.default_rng(seed)
    input_shape = onnx_model.get_inputs()[0].shape
    input_name = onnx_model.get_inputs()[0].name
    input_type = onnx_model.get_inputs()[0].type
    if "uint8" in input_type:
        sample = rng.integers(0, 255, size=input_shape, dtype=np.uint8)
    else:
        sample = rng.random(input_shape, dtype=np.float32)  # raw [0,1], host-preprocessed shape
    ref = onnx_model.run(None, {input_name: sample})[0].astype(np.float32).reshape(-1)

    print(f"[verify] inference job on '{device.name}' ...")
    ijob = hub.submit_inference_job(
        model=target, device=device,
        inputs={"input": [sample]},
        name="yowov3-verify",
    )
    out = ijob.download_output_data()
    if out is None:
        raise RuntimeError(f"inference produced no output: {ijob.url}")
    dev = np.array(list(out.values())[0][0], dtype=np.float32).reshape(-1)

    if dev.shape != ref.shape:
        print(f"[verify] !! shape mismatch dev{dev.shape} vs ref{ref.shape}")
    max_abs = float(np.max(np.abs(dev - ref)))
    mse = float(np.mean((dev - ref) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * np.log10((1.0 ** 2) / mse)
    print(f"[verify] max-abs-diff={max_abs:.6g}  psnr={psnr:.2f} dB")
    ok = max_abs < 1e-2
    print(f"[verify] {'OK' if ok else 'DRIFT — inspect BGR/layout/quant'} ({ijob.url})")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compile and profile YOWOv3 on Snapdragon device.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yowov3.onnx", help="Path to the ONNX model")
    ap.add_argument("--out-dir", default="models/qnn", help="Output directory for compiled binary")
    ap.add_argument("--device", default="Snapdragon X Elite CRD", help="Device name on QAI Hub")
    ap.add_argument("--quantize", action="store_true", help="add INT8 quantize step (default fp16)")
    ap.add_argument("--calib-dir", help="image dir for quantize calibration")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = hub.Device(args.device)
    onnx_path = Path(args.onnx_path)

    if not onnx_path.exists():
        sys.exit(f"Error: ONNX model file does not exist: {onnx_path}")

    model_src = onnx_path
    if args.quantize:
        model_src = quantize(hub, onnx_path, args.calib_dir)

    cjob, target = compile_qnn(hub, model_src, device)

    # download compiled binary
    bin_path = out_dir / "yowov3.bin"
    try:
        target.download(str(bin_path))
        print(f"[compile] saved {bin_path}")
    except Exception as e:  # download is best-effort; job/model still on hub
        print(f"[compile] download skipped: {e}")

    if not args.no_profile:
        profile(hub, target, device)

    if not args.no_verify:
        onnx_model = Ort.InferenceSession(str(onnx_path))
        verify(hub, target, onnx_model, device, seed=args.seed)

if __name__ == "__main__":
    main()
