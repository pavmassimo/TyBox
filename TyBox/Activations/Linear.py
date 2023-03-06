import numpy as np
import math

from .Activation import Activation


class Linear(Activation):
    @staticmethod
    def activate(self, vector):
        return vector

    @staticmethod
    def derivative(self, vector):
        return np.ones(len(vector))
