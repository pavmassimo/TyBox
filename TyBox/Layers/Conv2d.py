from .Layer import Layer


class Conv2dLayer(Layer):
    def __init__(self, layer_keras, layer_name):
        super().__init__(layer_keras, layer_name)
        self.evaluate_layer()

    def evaluate_layer(self):
        layer = self.layer_keras

        filter_rows = layer.weights[0].shape[0]
        filter_columns = layer.weights[0].shape[1]
        filter_channels = layer.weights[0].shape[2]

        n_filters = layer.weights[0].shape[3]

        input_rows = layer.input_shape[1]
        input_columns = layer.input_shape[2]
        input_channels = layer.input_shape[3]

        stride = layer.strides

        n_macc = n_filters * (filter_rows * filter_columns + 1) * filter_channels * \
                 (((input_rows - filter_rows) / stride[0]) + 1) * \
                 (((input_columns - filter_columns) / stride[1]) + 1)

        n_flops = ((input_rows - filter_rows) / stride[0] + 1) * \
                  ((input_columns - filter_columns) / stride[1] + 1) * \
                  (input_channels * (2 * filter_rows * filter_channels + 1) * n_filters)

        self.operations = {
            'flops': n_flops,
            'maccs': n_macc,
            'divisions': 0,
            'sums': 0,
            'comparisons': 0
        }
