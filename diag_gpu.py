import time
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("free/total MB:", [v // 2**20 for v in torch.cuda.mem_get_info()])
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(50):
        a @ a
    torch.cuda.synchronize()
    print("matmul TFLOPS:", 50 * 2 * 4096 ** 3 / (time.time() - t) / 1e12)
