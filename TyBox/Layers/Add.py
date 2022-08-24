from Layer import Layer


class AddLayer(Layer):
    def __init__(self, layer_keras, layer_name):
        super().__init__(layer_keras, layer_name)
        self.evaluate_layer()

    def evaluate_layer(self):
        layer = self.layer_keras

        n_sums = 1
        for i in layer.input_shape[0]:
            if i is not None:
                n_sums *= i

        self.operations = {
            'flops': 0,
            'maccs': 0,
            'divisions': 0,
            'sums': n_sums,
            'comparisons': 0
        }