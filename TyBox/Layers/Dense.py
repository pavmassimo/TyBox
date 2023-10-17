from .Layer import Layer


class DenseLayer(Layer):
    def __init__(self, layer_keras, layer_name):
        super().__init__(layer_keras, layer_name)
        self.evaluate_layer()

    def evaluate_layer(self):
        layer = self.layer_keras

        n_inputs = layer.weights[0].shape[0]
        n_outputs = layer.weights[0].shape[1]
        n_macc = n_outputs * (n_inputs + 1)
        n_flops = 2 * n_inputs * n_outputs

        self.operations = {
            'flops': n_flops,
            'maccs': n_macc,
            'divisions': 0,
            'sums': 0,
            'comparisons': 0
        }