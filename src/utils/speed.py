import torch, os
def speed_setup(channels_last: bool, cudnn_benchmark: bool):
    # Enable TF32 fast paths on Ampere+ / Blackwell
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    # Prefer Flash and mem-efficient SDPA when available
    try:
        torch.nn.attention.sdpa_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
    except Exception:
        pass
    torch.set_float32_matmul_precision("high")  # numerically safe speedup
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)  # algo autotune for fixed shapes
    # Optional: model.to(memory_format=torch.channels_last) per-module
