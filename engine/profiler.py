import time
import torch

class Profiler:
    def measure(self, fn, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        out = fn(*args)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        return {
            "output": out,
            "latency_ms": (end - start) * 1000
        }
