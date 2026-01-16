class InferenceEngine:
    def __init__(self, backend, batcher=None, profiler=None):
        self.backend = backend
        self.batcher = batcher
        self.profiler = profiler
        if self.batcher:
            self.batcher.start(self.backend.infer)

    def run(self, inputs):
        if self.batcher:
            return self.batcher.add(inputs)
        if self.profiler:
            return self.profiler.measure(self.backend.infer, inputs)
        return self.backend.infer(inputs)
