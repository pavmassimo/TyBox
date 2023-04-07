import numpy as np


class Buffer:
    def __init__(self, size_inputs: int, size_outputs: int, size_buffer: int):
        self.size_buffer = size_buffer
        self.buffer = np.zeros((size_buffer, size_inputs))
        self.labels = np.zeros((size_buffer, size_outputs))
        self.is_full = False
        self.pointer = 0

    def push(self, sample, label):
        if self.pointer < self.size_buffer:
            self.buffer[self.pointer] = sample
            self.labels[self.pointer] = label
            self.pointer += 1
        else:
            self.is_full = True
            self.buffer[0] = sample
            self.labels[0] = label
            self.pointer = 1

    def get_buffer_samples(self):
        return self.buffer[:(max(self.pointer, self.is_full * self.size_buffer))]

    def get_buffer_labels(self):
        return self.labels[:(max(self.pointer, self.is_full * self.size_buffer))]
