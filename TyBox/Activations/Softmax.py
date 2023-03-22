import numpy as np
import math

from .Activation import Activation


class Softmax(Activation):
    @staticmethod
    def activate(vector):
        e = np.exp(vector)
        return e / e.sum()

    """vector assumed to be already activated"""
    @staticmethod
    def derivative(vector):
        s = np.array(vector)
        si_sj = -s * s.reshape(len(vector), 1)
        s_der = np.diag(s) + si_sj
        return s_der

    """vector assumed to be already activated"""
    def calculate_output_delta(self, vector, target):
        s_der = self.derivative(vector)
        output_errors = target - vector

        tmp = s_der @ output_errors

        return tmp

    #check this function
    def calculate_hidden_delta(self, vector, weights_vector, successive_layer_delta):
        print("hidden deltas for softmax not implemented yet")
        pass
