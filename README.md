# minimal-inference-engine
This project is a minimal LLM inference engine built from scratch with a systems-first mindset.<br>

It focuses exclusively on inference — not training — and is designed to
explore real-world performance tradeoffs such as latency vs throughput,
CPU vs GPU execution, and dynamic batching behavior.

The engine supports multiple backends including PyTorch, ONNX Runtime,
and TensorRT, and includes profiling tools and benchmarks to measure
end-to-end inference performance.

*No training code is included by design.*

## Batch size benchmark (what it shows)
We run a controlled benchmark that sends many single-sample requests and varies only the batch size. For each batch size, we measure:
- Throughput (requests/sec)
- Average latency
- P50 and P95 latency (median vs slow tail)

### How to run
```
python3 01_minimal-inference-engine/benchmarks/batch_size_bench.py \
  --requests 1024 --batch-sizes 1,2,4,8,16 --trials 10
```

This writes a plot to `01_minimal-inference-engine/result/batch_size_bench.png` showing:
- Throughput changes by batch size
- Average latency changes by batch size
- P50/P95 latency changes by batch size

![Batch size benchmark plot](01_minimal-inference-engine/result/batch_size_bench.png)

The takeaway is that batching has a “sweet spot.” Too small means poor throughput; too large increases waiting time. In our runs, mid-sized batches (like 4–8) typically balance throughput and latency best.

### Why the curves look this way
- At batch size 1, throughput is lowest because each request pays the full per-request overhead (scheduling, kernel launches, and framework bookkeeping), so the hardware is underutilized. Latency and tail latency (P50/P95) are highest because there is no opportunity to amortize those fixed costs across multiple requests.
- As batch size grows from 1 to a moderate value, throughput increases because fixed overhead is shared across more tokens and the accelerator stays busier. Average latency often drops in this range because the work per request becomes more efficient even though requests may wait briefly to form a batch.
- Tail latency (P50/P95) tends to rise as batch size gets large because requests spend more time waiting in the queue to be batched, and large batches take longer to execute, increasing variance and the slow tail.
- Throughput does not grow forever: it eventually saturates when the device is fully utilized, and can even fall if batches get so large that memory pressure, cache misses, or scheduling overheads dominate.
- Latency does not keep decreasing either: after the efficient-batching zone, both average and tail latency trend upward due to queueing delay and longer per-batch compute time.
