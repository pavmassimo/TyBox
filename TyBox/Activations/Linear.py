import numpy as np
import math

from Activation import Activation


class Linear(Activation):

    def activate(self, vector):
        return vector

    def derivative(self, vector):
        return np.ones(len(vector))
