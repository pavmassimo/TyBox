import numpy as np
import math

from .Activation import Activation


class Relu(Activation):

    def activate(self, vector):
        results = []
        for input_value in vector:
            result = input_value if input_value > 0 else 0
            results.append(result)
        results = np.array(results)
        return results

    """vector assumed to be already activated"""
    def derivative(self, vector):
        results = []
        for input_value in vector:
            result = 1 if input_value > 0 else 0
            results.append(result)

        results = np.array(results)
        return results
