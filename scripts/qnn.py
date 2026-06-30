import sys
import numpy as np
try:
    import qai_hub as hub
except ImportError:
    sys.exit("qai_hub missing — env should have qai-hub.")

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


def verify(hub, target, keras_model, spec: ModelSpec, device, seed: int = 0):
    """On-device vs keras numeric parity on one sample tensor."""
    rng = np.random.default_rng(seed)
    sample = rng.random(spec.input_shape, dtype=np.float32)  # raw [0,1], host-preprocessed shape
    ref = keras_model.predict(sample, verbose=0).astype(np.float32).reshape(-1)

    print(f"[verify] inference job on '{device.name}' ...")
    ijob = hub.submit_inference_job(
        model=target, device=device,
        inputs={"input": [sample]},
        name=f"{spec.model_name}-verify",
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
    if spec.model_name == "age":
        ages = np.arange(spec.output_len)
        print(f"[verify] age dev={float((dev*ages).sum()):.2f}  ref={float((ref*ages).sum()):.2f}")
    else:
        print(f"[verify] argmax dev={int(dev.argmax())}  ref={int(ref.argmax())}")
    ok = max_abs < 1e-2
    print(f"[verify] {'OK' if ok else 'DRIFT — inspect BGR/layout/quant'} ({ijob.url})")
    return ok