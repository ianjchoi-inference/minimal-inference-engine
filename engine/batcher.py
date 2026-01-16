import queue
import time
import threading
import torch

class DynamicBatcher:
    def __init__(self, max_batch_size=8, timeout_ms=10):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker = None
        self._infer_fn = None

    def add(self, x):
        event = threading.Event()
        self.queue.put((x, event))
        event.wait()
        return event.result

    def start(self, infer_fn):
        if self._worker and self._worker.is_alive():
            return
        self._infer_fn = infer_fn
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def stop(self, drain=False):
        self._stop_event.set()
        if drain and self._infer_fn:
            self.flush(self._infer_fn)
        if self._worker:
            self._worker.join(timeout=1)

    def _run(self):
        while not self._stop_event.is_set():
            if self._infer_fn is None:
                time.sleep(self.timeout_ms / 1000)
                continue
            self.flush(self._infer_fn)

    def flush(self, infer_fn):
        batch = []
        events = []

        while len(batch) < self.max_batch_size:
            try:
                x, ev = self.queue.get(timeout=self.timeout_ms / 1000)
                batch.append(x)
                events.append(ev)
            except queue.Empty:
                break

        if not batch:
            return

        inputs = torch.cat(batch, dim=0)
        outputs = infer_fn(inputs)

        for ev, out in zip(events, outputs):
            ev.result = out
            ev.set()
