import numpy as np


class PyBoxBuffer:
    def __init__(self, size_inputs: int, size_outputs: int, size_buffer: int):
        self.buffer = np.zeros((size_buffer, size_inputs))
        self.labels = np.zeros((size_buffer, size_outputs))
        self.is_full = False
