import numpy as np
import math

from .Activation import Activation


class Sigmoid(Activation):
    @staticmethod
    def activate(vector):
        results = []
        for input_value in vector:
            try:
                result = 1 / (1 + math.exp(-input_value))
            except Exception as e:
                print(input_value, e)
                raise Exception
            results.append(result)
        results = np.array(results)
        return results

    """vector assumed to be already activated"""

    @staticmethod
    def derivative(vector):
        results = []
        for input_value in vector:
            result = input_value * (1 - input_value)
            results.append(result)

        results = np.array(results)
        return results
