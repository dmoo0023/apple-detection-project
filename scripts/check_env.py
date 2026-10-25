"""Stage 0 exit check: print key package versions and confirm the GPU is usable by PyTorch.

Run with: .venv\\Scripts\\python.exe scripts\\check_env.py
"""

import sys
import platform


def main() -> int:
    print("Python:", sys.version.split()[0], "(" + platform.platform() + ")")

    import torch
    import torchvision
    import ultralytics
    import cv2
    import numpy
    import scipy
    import pandas
    import yaml  # pyyaml

    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("ultralytics:", ultralytics.__version__)
    print("opencv-python:", cv2.__version__)
    print("numpy:", numpy.__version__)
    print("scipy:", scipy.__version__)
    print("pandas:", pandas.__version__)
    print("pyyaml:", yaml.__version__)

    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)
    if not cuda_available:
        print("GPU CHECK FAILED: torch.cuda.is_available() is False.")
        return 1

    device_name = torch.cuda.get_device_name(0)
    print("GPU device:", device_name)
    print("CUDA build version (torch):", torch.version.cuda)

    # Run a small matmul on the GPU to confirm the kernels actually execute
    # (is_available() can be True even if the architecture isn't supported by the build).
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    print("GPU matmul check: OK, result shape", tuple(c.shape))

    print("\nGPU CHECK PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
