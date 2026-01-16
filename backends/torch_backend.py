import torch
from engine.backend_base import Backend

class TorchBackend(Backend):
    def __init__(self, device="cpu"):
        self.device = device
        self.model = torch.nn.Linear(10, 2).to(device)
        self.model.eval()

    def infer(self, inputs):
        inputs = inputs.to(self.device)
        with torch.no_grad():
            return self.model(inputs)

    def load_model(self, path):
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.model.eval()
