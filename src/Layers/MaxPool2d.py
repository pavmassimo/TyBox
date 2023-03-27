from .Layer import Layer


class MaxPool2dLayer(Layer):
    def __init__(self, layer_keras, layer_name):
        super().__init__(layer_keras, layer_name)
        self.evaluate_layer()

    def evaluate_layer(self):
        layer = self.layer_keras

        # cost is number of comparisons
        output_w = layer.output_shape[1]
        output_h = layer.output_shape[2]
        output_c = layer.output_shape[3]
        cost = output_w * output_h * output_c * layer.pool_size[0] * layer.pool_size[1]

        self.operations = {
            'flops': 0,
            'maccs': 0,
            'divisions': 0,
            'sums': 0,
            'comparisons': cost
        }
