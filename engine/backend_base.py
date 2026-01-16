from abc import ABC, abstractmethod

class Backend(ABC):
    @abstractmethod
    def infer(self, inputs):
        """Run inference"""
        pass

    @abstractmethod
    def load_model(self, path: str):
        """Load model weights"""
        pass
