
class Layer:
    def __init__(self, layer_keras, layer_name, weight_dim=32, act_dim=32):
        self.layer_keras = layer_keras
        self.layer_name = layer_name
        self.weight_dim = weight_dim
        self.act_dim = act_dim

        self.parameters = 0

        for n in layer_keras.weights:
            p = 1
            for nn in n.shape:
                p *= nn
            self.parameters += p

        self.activations = None

        output_shape = layer_keras.output_shape[1:]
        output_activations = 1
        for dimension in output_shape:
            output_activations *= dimension

        self.activations = output_activations

        self.operations = {
            'flops': None,
            'maccs': None,
            'divisions': None,
            'sums': None,
            'comparisons': None
        }

    def get_sizes(self):
        return {'param': self.parameters,
                'act':self.activations}

    def get_operations(self):
        return self.operations

    def get_name(self):
        return self.layer_name

    def evaluate_layer(self):
        pass
