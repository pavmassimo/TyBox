import numpy as np
import math
from .create_a import create_activation

class Model:
    def __init__(self, layer_widths, activation_functions, len_buff):
        self.inputs = layer_widths[0]
        self.outputs = layer_widths[-1]
        self.layers = [[0 for ii in range(layer_widths[i])] for i in range(len(layer_widths))]
        self.weights = [np.random.rand(layer_widths[i], layer_widths[i + 1]) for i in range(len(layer_widths) - 1)]
        self.biases = [np.empty((layer_widths[i + 1])) for i in range(len(layer_widths) - 1)]
        self.activation_functions = []

        for activation_function in activation_functions:
            self.activation_functions.append(create_activation(activation_function))

        self.buffer = np.zeros((len_buff, self.inputs))
        self.labels = np.zeros((len_buff, self.outputs))
        self.buffer_full = False
        self.buffer_pointer = 0

        self.accuracies = np.zeros(len_buff)
        self.accuracies_pointer = 0
        self.accuracies_full = False

        self.accuracy = 0
        self.max_acc = 0
        self.elements_when_max = 0
        self.alert_pointer = None

        self.lr = 0.01

    def read_weights(self, files):
        # print('read weights', len(files), len(self.layers) - 1)
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
        # print(input_value.shape[0], self.inputs)
        assert input_value.shape[0] == self.inputs
        self.layers[0] = [i for i in input_value]
        for i in range(len(self.layers) - 1):
            for node_index in range(len(self.layers[i + 1])):
                self.layers[i + 1][node_index] = self.biases[i][node_index]
                for input_node_index in range(len(self.layers[i])):
                    self.layers[i + 1][node_index] += self.layers[i][input_node_index] * \
                                                      self.weights[i][input_node_index][node_index]
            self.layers[i + 1] = self.activation_functions[i].activate(self.layers[i + 1])

    def execute_backprop(self, target, lr):
        output_deltas = self.calculate_output_delta(target)
        self.update_weights_and_biases_of_layer(len(self.layers) - 1, output_deltas, lr)
        successive_layer_deltas = output_deltas
        # print('a')
        for layer_index in range(len(self.layers) - 2, 0, -1):
            layer_deltas = self.calculate_hidden_delta(layer_index, successive_layer_deltas)
            # print(layer_deltas)
            self.update_weights_and_biases_of_layer(layer_index, layer_deltas, lr)
            successive_layer_deltas = layer_deltas

    def calculate_output_delta(self, target):
        assert len(target) == self.outputs
        delta_output_list = self.activation_functions[-1].calculate_output_delta(self.layers[-1], target)

        assert len(delta_output_list) == self.outputs
        return delta_output_list

    def calculate_hidden_delta(self, layer_index, successive_layer_delta):
        assert 0 < layer_index < (len(self.layers) - 1)
        assert len(successive_layer_delta) == len(self.layers[layer_index + 1])
        delta_hidden_list = self.activation_functions[layer_index].calculate_hidden_delta(self.layers[layer_index + 1],
                                                                                          self.weights[layer_index],
                                                                                          successive_layer_delta)
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
            calculated_label = list(self.layers[-1]).index(max(self.layers[-1]))
            if target_label == calculated_label:
                score += 1
        return score/len(inputs)

    def evaluate_one(self, input_data, target):
        self.execute_forward_pass(input_data)
        calculated_label = self.layers[-1].index(max(self.layers[-1]))
        target_label = list(target).index(max(target))
        if calculated_label == target_label:
            return 1
        return 0

    def train(self):
        for input_index in range(self.buffer_pointer):
            self.execute_forward_pass(self.buffer[input_index])
            self.execute_backprop(self.labels[input_index], self.lr)

    def push_datum(self, datum, target):
        for i in range(len(datum)):
            self.buffer[self.buffer_pointer][i] = datum[i]
        self.labels[self.buffer_pointer] = target
        self.buffer_pointer = self.buffer_pointer + 1

        if self.buffer_pointer >= len(self.buffer):
            self.buffer_pointer = 0

    def push_data(self, data, targets):
        for input_index in range(len(data)):
            self.push_datum(data[input_index], targets[input_index])

    def set_lr(self, lr):
        self.lr = lr

    def push_result(self, result):

        self.accuracies[self.accuracies_pointer] = result
        self.accuracies_pointer_pointer += 1

        if self.accuracies_pointer >= len(self.accuracies):
            self.accuracies_pointer = 0
            self.accuracies_full = True

    def evaluate_push_and_train(self, datum, target):
        result = self.evaluate_one(datum, target)
        self.push_result(result)
        self.push_datum(datum, target)
        self.train()
        self.accuracy = self.running_average_accuracy()

        if self.accuracy > self.max_acc:
            self.max_acc = self.accuracy
            self.elements_when_max = max(self.accuracies_pointer, len(self.accuracies) * self.accuracies_full)

        self.abrupt_cd_detector()
        return self.accuracy

    def running_average_accuracy(self):
        if self.accuracies_full:
            return np.mean(self.accuracies)
        else:
            return np.mean(self.accuracies[:self.accuracies_pointer])

    def abrupt_cd_detector(self):

        self.accuracy = self.running_average_accuracy()

        if self.accuracy > self.max_acc:
            self.max_acc = self.accuracy
            self.elements_when_max = max(self.accuracies_pointer, len(self.accuracies) * self.accuracies_full)

        p_i = 1-self.accuracy
        p_min = 1-self.max_acc

        delta_i = math.sqrt((p_i * (1-p_i)) / max(self.accuracies_pointer, len(self.accuracies) * self.accuracies_full))
        delta_min = math.sqrt((p_i * (1-p_i)) / self.elements_when_max)

        if (p_i + delta_i > p_min + 3 * delta_min and self.alert_pointer is not None):
            print("Concept drift!")
            if self.buffer_pointer >= self.alert_pointer:
                diff = self.buffer_pointer - self.alert_pointer
                self.buffer[:diff] = self.buffer[self.alert_pointer: self.buffer_pointer]

            else:
                diff = self.buffer_pointer + len(self.buffer) - self.alert_pointer
                self.buffer[:diff] = np.concatenate(self.buffer[self.alert_pointer:], self.buffer[:self.buffer_pointer])

            self.buffer_pointer = diff
            self.buffer_full = False

            self.accuracies = np.zeros(self.len_buff)
            self.accuracies_pointer = 0
            self.accuracies_full = False

            self.alert_pointer = None
            self.accuracy = 0
            self.max_acc = 0
            self.elements_when_max = 0
        if (p_i + delta_i > p_min + 2 * delta_min):
            print("alert")
            self.alert_pointer = self.buffer_pointer
        else:
            self.alert_pointer = None

