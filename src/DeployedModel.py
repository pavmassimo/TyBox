from PyBoxBuffer import PyBoxBuffer
from src import PyBoxModel


class DeployedModel:
    def __init__(self, model: PyBoxModel, buffer: PyBoxBuffer):
        self.model = model
        self.buffer = buffer
