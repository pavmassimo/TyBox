from Layer import Layer


class AvgPool2dLayer(Layer):
    def __init__(self, layer_keras, layer_name):
        super().__init__(layer_keras, layer_name)
        self.evaluate_layer()

    def evaluate_layer(self):
        layer = self.layer_keras

        output_w = layer.output_shape[1]
        output_h = layer.output_shape[2]
        output_c = layer.output_shape[3]
        output_buffer_size = output_w * output_h * output_c
        sums_per_cell = layer.pool_size[0] * layer.pool_size[1]

        cost_sum = sums_per_cell * output_buffer_size
        cost_division = output_buffer_size

        self.operations = {
            'flops': 0,
            'maccs': 0,
            'divisions': cost_division,
            'sums': cost_sum,
            'comparisons': 0
        }
