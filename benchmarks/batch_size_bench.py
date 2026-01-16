import argparse
import os
import statistics
import threading
import time
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backends.torch_backend import TorchBackend
from engine.batcher import DynamicBatcher
from engine.engine import InferenceEngine


def _percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((pct / 100.0) * (len(values) - 1)))
    return values[idx]


def run_batch_size_bench(
    batch_sizes,
    num_requests,
    timeout_ms,
    device,
    warmup,
    save_plot,
    trials,
    arrival_ms,
):
    rows = []
    for batch_size in batch_sizes:
        trial_metrics = []
        for _ in range(trials):
            metrics = _run_once(
                batch_size=batch_size,
                num_requests=num_requests,
                timeout_ms=timeout_ms,
                device=device,
                warmup=warmup,
                arrival_ms=arrival_ms,
            )
            trial_metrics.append(metrics)

        throughput_vals = [m["throughput_rps"] for m in trial_metrics]
        avg_vals = [m["lat_ms_avg"] for m in trial_metrics]
        p50_vals = [m["lat_ms_p50"] for m in trial_metrics]
        p95_vals = [m["lat_ms_p95"] for m in trial_metrics]

        rows.append(
            {
                "batch": batch_size,
                "throughput_rps_mean": statistics.mean(throughput_vals),
                "throughput_rps_std": _stddev(throughput_vals),
                "lat_ms_avg_mean": statistics.mean(avg_vals),
                "lat_ms_avg_std": _stddev(avg_vals),
                "lat_ms_p50_mean": statistics.mean(p50_vals),
                "lat_ms_p50_std": _stddev(p50_vals),
                "lat_ms_p95_mean": statistics.mean(p95_vals),
                "lat_ms_p95_std": _stddev(p95_vals),
            }
        )

    _print_table(rows)
    if save_plot:
        _save_plot(rows, save_plot, num_requests, trials, batch_sizes)


def _run_once(batch_size, num_requests, timeout_ms, device, warmup, arrival_ms):
    backend = TorchBackend(device=device)
    batcher = DynamicBatcher(max_batch_size=batch_size, timeout_ms=timeout_ms)
    engine = InferenceEngine(backend, batcher=batcher)

    if warmup > 0:
        for _ in range(warmup):
            _ = engine.run(torch.randn(1, 10))

    latencies_ms = []
    lock = threading.Lock()

    def task():
        x = torch.randn(1, 10)
        start = time.perf_counter()
        _ = engine.run(x)
        end = time.perf_counter()
        with lock:
            latencies_ms.append((end - start) * 1000)

    start_all = time.perf_counter()
    threads = []
    for _ in range(num_requests):
        t = threading.Thread(target=task)
        threads.append(t)
        t.start()
        if arrival_ms > 0:
            time.sleep(arrival_ms / 1000)
    for t in threads:
        t.join()
    end_all = time.perf_counter()

    batcher.stop(drain=True)

    total_s = end_all - start_all
    throughput = num_requests / total_s if total_s > 0 else 0.0
    p50 = _percentile(latencies_ms, 50)
    p95 = _percentile(latencies_ms, 95)
    avg = statistics.mean(latencies_ms) if latencies_ms else 0.0

    return {
        "throughput_rps": throughput,
        "lat_ms_avg": avg,
        "lat_ms_p50": p50,
        "lat_ms_p95": p95,
    }


def _stddev(values):
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _print_table(rows):
    if not rows:
        return
    headers = [
        ("batch", "Batch"),
        ("throughput_rps", "Throughput(rps)"),
        ("lat_ms_avg", "Avg Lat(ms)"),
        ("lat_ms_p50", "P50 Lat(ms)"),
        ("lat_ms_p95", "P95 Lat(ms)"),
    ]
    formatted_rows = []
    for row in rows:
        formatted_rows.append(
            [
                str(row["batch"]),
                _format_mean_std(
                    row["throughput_rps_mean"], row["throughput_rps_std"]
                ),
                _format_mean_std(row["lat_ms_avg_mean"], row["lat_ms_avg_std"]),
                _format_mean_std(row["lat_ms_p50_mean"], row["lat_ms_p50_std"]),
                _format_mean_std(row["lat_ms_p95_mean"], row["lat_ms_p95_std"]),
            ]
        )

    widths = []
    for idx, (_, title) in enumerate(headers):
        max_len = len(title)
        for row in formatted_rows:
            max_len = max(max_len, len(row[idx]))
        widths.append(max_len)

    header_line = " | ".join(
        title.ljust(widths[idx]) for idx, (_, title) in enumerate(headers)
    )
    sep_line = "-+-".join("-" * w for w in widths)
    print(header_line)
    print(sep_line)
    for row in formatted_rows:
        print(" | ".join(row[idx].rjust(widths[idx]) for idx in range(len(headers))))


def _format_mean_std(mean, std):
    return f"{mean:.2f}+-{std:.2f}"


def _save_plot(rows, output_path, num_requests, trials, batch_sizes):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            "matplotlib is required for plotting. Install it or rerun without --plot."
        )

    if not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)

    batches = [row["batch"] for row in rows]
    positions = list(range(len(batches)))
    throughput = [row["throughput_rps_mean"] for row in rows]
    throughput_std = [row["throughput_rps_std"] for row in rows]
    avg = [row["lat_ms_avg_mean"] for row in rows]
    avg_std = [row["lat_ms_avg_std"] for row in rows]
    p50 = [row["lat_ms_p50_mean"] for row in rows]
    p50_std = [row["lat_ms_p50_std"] for row in rows]
    p95 = [row["lat_ms_p95_mean"] for row in rows]
    p95_std = [row["lat_ms_p95_std"] for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    batch_label = ",".join(str(b) for b in batch_sizes)
    fig.suptitle(
        f"requests={num_requests}, trials={trials}, batches={batch_label}",
        fontsize=10,
    )

    axes[0].bar(
        positions, throughput, color="tab:blue", yerr=throughput_std, capsize=3
    )
    axes[0].set_title("Throughput")
    axes[0].set_xlabel("Batch size")
    axes[0].set_xticks(positions, [str(b) for b in batches])
    axes[0].set_ylabel("RPS")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[1].errorbar(
        positions, avg, yerr=avg_std, marker="o", label="Avg", capsize=3
    )
    axes[1].set_title("Latency (ms)")
    axes[1].set_xlabel("Batch size")
    axes[1].set_xticks(positions, [str(b) for b in batches])
    axes[1].set_ylabel("Milliseconds")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(loc="best")

    bar_width = 0.35
    offsets = [p - (bar_width / 2.0) for p in positions]
    offsets2 = [p + (bar_width / 2.0) for p in positions]
    axes[2].bar(
        offsets,
        p50,
        width=bar_width,
        color="#1f77b4",
        yerr=p50_std,
        capsize=3,
        label="P50",
    )
    axes[2].bar(
        offsets2,
        p95,
        width=bar_width,
        color="#ff7f0e",
        yerr=p95_std,
        capsize=3,
        label="P95",
    )
    axes[2].set_title("P50 vs P95")
    axes[2].set_xlabel("Batch size")
    axes[2].set_xticks(positions, [str(b) for b in batches])
    axes[2].set_ylabel("Milliseconds")
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[2].legend(loc="best")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch size benchmark for the minimal inference engine."
    )
    parser.add_argument("--batch-sizes", default="1,2,4,8", help="Comma list.")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--timeout-ms", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--arrival-ms",
        type=int,
        default=0,
        help="Delay between request starts to simulate non-burst traffic.",
    )
    parser.add_argument(
        "--plot",
        default=os.path.join("result", "batch_size_bench.png"),
        help="Save a PNG plot to this path.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable saving the PNG plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    run_batch_size_bench(
        batch_sizes=batch_sizes,
        num_requests=args.requests,
        timeout_ms=args.timeout_ms,
        device=args.device,
        warmup=args.warmup,
        save_plot=None if args.no_plot else args.plot,
        trials=args.trials,
        arrival_ms=args.arrival_ms,
    )


if __name__ == "__main__":
    main()
