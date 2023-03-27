import numpy as np
import math

from .Activation import Activation


class Linear(Activation):
    @staticmethod
    def activate(vector):
        return vector

    @staticmethod
    def derivative(vector):
        return np.ones(len(vector))
