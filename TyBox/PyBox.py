import numpy as np

from TyBox.Activations import Sigmoid, Softmax


class Model:
    def __init__(self, layer_widths):
        self.inputs = layer_widths[0]
        self.outputs = layer_widths[-1]
        self.layers = [[0 for _ in range(layer_widths[i])] for i in range(len(layer_widths))]
        self.weights = [np.random.rand(layer_widths[i], layer_widths[i + 1]) for i in range(len(layer_widths) - 1)]
        self.biases = [np.empty((layer_widths[i + 1])) for i in range(len(layer_widths) - 1)]
        self.learning_rate = 0

    def set_lr(self, lr: int) -> None:
        self.learning_rate = lr

    def read_weights(self, files):
        number_of_connection_groups = len(self.layers) - 1
        assert len(files) == number_of_connection_groups
        for layer_index in range(number_of_connection_groups):
            with open(files[layer_index], "r") as file:
                file.readline()
                for i in range(len(self.layers[layer_index])):
                    line = file.readline().split(' ')[:-1]
                    weights = []
                    for weight in line:
                        weights.append(float(weight))
                    self.weights[layer_index][i] = weights
                file.readline()
                line = file.readline().split(' ')[:-1]
                biases = []
                for bias in line:
                    biases.append(float(bias))
                self.biases[layer_index] = biases

    def execute_forward_pass(self, input_value):
        assert len(input_value.shape) == 1
        assert input_value.shape[0] == self.inputs
        self.layers[0] = [i for i in input_value]
        for i in range(len(self.layers) - 1):
            for node_index in range(len(self.layers[i + 1])):
                self.layers[i + 1][node_index] = self.biases[i][node_index]
                for input_node_index in range(len(self.layers[i])):
                    self.layers[i + 1][node_index] += self.layers[i][input_node_index] * \
                                                      self.weights[i][input_node_index][node_index]
                if i < len(self.layers) - 2:
                    self.layers[i + 1][node_index] = Sigmoid.activate(self.layers[i + 1][node_index])
        act = np.array(self.layers[-1])
        act = Softmax.activate(act)
        for i in range(len(self.layers[-1])):
            self.layers[-1][i] = act[i]

    def execute_backprop(self, target, lr):
        output_deltas = self.calculate_output_delta(target)
        self.update_weights_and_biases_of_layer(len(self.layers) - 1, output_deltas, lr)
        successive_layer_deltas = output_deltas
        for layer_index in range(len(self.layers) - 2, 0, -1):
            layer_deltas = self.calculate_hidden_delta(layer_index, successive_layer_deltas)
            self.update_weights_and_biases_of_layer(layer_index, layer_deltas, lr)
            successive_layer_deltas = layer_deltas

    def calculate_output_delta(self, target):
        assert len(target) == self.outputs
        delta_output_list = []
        for output_index in range(self.outputs):
            delta = Sigmoid.derivative_scalar(self.layers[-1][output_index]) * (
                    target[output_index] - self.layers[-1][output_index])
            delta_output_list.append(delta)
        assert len(delta_output_list) == self.outputs
        return delta_output_list

    def calculate_hidden_delta(self, layer_index, successive_layer_delta):
        assert 0 < layer_index < (len(self.layers) - 1)
        assert len(successive_layer_delta) == len(self.layers[layer_index + 1])
        delta_hidden_list = []
        for node_index in range(len(self.layers[layer_index])):
            sum_of_succ_layer_deltas = 0
            for next_layer_node_index in range(len(self.layers[layer_index + 1])):
                sum_of_succ_layer_deltas += successive_layer_delta[next_layer_node_index] \
                                            * self.weights[layer_index][node_index][next_layer_node_index]
            delta = Sigmoid.derivative(self.layers[layer_index][node_index]) * sum_of_succ_layer_deltas
            delta_hidden_list.append(delta)
        assert len(delta_hidden_list) == len(self.layers[layer_index])
        return delta_hidden_list

    # do not use negative indices for layer index, they work for the layers but not for the weights
    def update_weights_and_biases_of_layer(self, layer_index, layer_deltas, lr):
        assert 0 < layer_index < len(self.layers)
        assert len(layer_deltas) == len(self.layers[layer_index])
        for node_index in range(len(self.layers[layer_index])):
            for previous_layer_node_index in range(len(self.layers[layer_index - 1])):
                self.weights[layer_index - 1][previous_layer_node_index][node_index] += lr * \
                                                                                        layer_deltas[node_index] * \
                                                                                        self.layers[layer_index - 1][
                                                                                            previous_layer_node_index]
            self.biases[layer_index - 1][node_index] += lr * layer_deltas[node_index]

    def evaluate(self, inputs, targets):
        score = 0
        assert len(inputs) == len(targets)
        for input_index in range(len(inputs)):
            self.execute_forward_pass(inputs[input_index])
            target_label = list(targets[input_index]).index(max(targets[input_index]))
            calculated_label = self.layers[-1].index(max(self.layers[-1]))
            if target_label == calculated_label:
                score += 1
        return score / len(inputs)

    def evaluate_one(self, input_data, target):
        self.execute_forward_pass(input_data)
        calculated_label = self.layers[-1].index(max(self.layers[-1]))
        target_label = list(target).index(max(target))
        if calculated_label == target_label:
            return True
        return False

    def train(self, inputs, targets, lr):
        for input_index in range(len(inputs)):
            self.execute_forward_pass(inputs[input_index])
            self.execute_backprop(targets[input_index], lr)
