# temporarily adding to build some tests, will transition to updated implementations of the activations
import math

import numpy as np


def sigmoid(input_value):
    # if input_value < -10:
    #     return sigmoid(-10)
    # if input_value > 10:
    #     return sigmoid(10)
    try:
        result = 1 / (1 + math.exp(-input_value))
    except Exception as e:
        print(input_value, e)
        raise Exception
    return result


def sigmoid_derivative(input_value):
    result = input_value * (1 - input_value)
    return result


def softmax(vector):
    e = np.exp(vector)
    return e / e.sum()
