import threading
import torch

from engine.engine import InferenceEngine
from engine.batcher import DynamicBatcher
from backends.torch_backend import TorchBackend

backend = TorchBackend()
batcher = DynamicBatcher(max_batch_size=4, timeout_ms=5)
engine = InferenceEngine(backend, batcher=batcher)

results = [None] * 4

def run_one(idx):
    inputs = torch.randn(1, 10)
    results[idx] = engine.run(inputs)

threads = []
for i in range(4):
    t = threading.Thread(target=run_one, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

for out in results:
    print(out)
