from .Layer import Layer


class ReluLayer(Layer):
    def __init__(self, layer_keras, layer_name):
        super().__init__(layer_keras, layer_name)
        self.evaluate_layer()

    def evaluate_layer(self):
        layer = self.layer_keras

        buffer_size = 1
        for dim in layer.input_shape:
            if dim is not None:
                buffer_size *= dim
        # print('cost: ', buffer_size, ' binary comparisons')

        self.operations = {
            'flops': 0,
            'maccs': 0,
            'divisions': 0,
            'sums': 0,
            'comparisons': buffer_size
        }
