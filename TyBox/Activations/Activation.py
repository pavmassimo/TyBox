from abc import abstractmethod, ABC

import numpy as np
import math


class Activation(ABC):
    @staticmethod
    def activate(vector):
        pass

    @staticmethod
    def derivative(vector):
        pass

    """vector assumed to be already activated"""

    def calculate_output_delta(self, vector, target):
        delta_output_list = self.derivative(vector) * (target - vector)

        return delta_output_list

    def calculate_hidden_delta(self, vector, weights_vector, successive_layer_delta):
        delta_hidden_list = []
        derivative = self.derivative(vector)
        for node_index in range(len(vector)):
            sum_of_succ_layer_deltas = np.sum(successive_layer_delta * weights_vector[node_index])
            delta = derivative[node_index] * sum_of_succ_layer_deltas
            delta_hidden_list.append(delta)
        return delta_hidden_list


# are these duplicates?
def sigmoid(input_value):
    try:
        result = 1 / (1 + math.exp(-input_value))
    except Exception as e:
        raise Exception(f"{input_value}, {e}")
    return result


def sigmoid_derivative(input_value):
    result = input_value * (1 - input_value)
    return result


def softmax(vector):
    e = np.exp(vector)
    return e / e.sum()


def softmax_derivative(softmax_out, output_errors):
    s = np.array(softmax_out)
    si_sj = -s * s.reshape(len(softmax_out), 1)
    s_der = np.diag(s) + si_sj
    tmp = s_der @ output_errors
    return tmp
