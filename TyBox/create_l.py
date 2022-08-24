import TyBox.Layers as l


def create_layer(layer):
    print("create_l")
    print(type(layer))
    layer_name = "{}".format(layer).split('.')[-1].split(' ')[0]
    if layer_name == 'Activation':
        layer_name = "{}".format(layer.activation).split(' ')[1]

    if layer_name == 'Conv2D' or layer_name == 'SeparableConv2D' or layer_name == 'DepthwiseConv2D':
        return l.Conv2dLayer(layer, layer_name)

    elif layer_name == 'Dense':
        return l.DenseLayer(layer, layer_name)

    elif layer_name == 'AveragePooling2D':
        return l.AvgPool2dLayer(layer, layer_name)

    elif layer_name == 'MaxPooling2D':
        return l.MaxPool2dLayer(layer, layer_name)

    elif layer_name == 'ReLU' or layer_name == 'LeakyReLU' or layer_name == 'relu':
        return l.ReluLayer(layer, layer_name)

    elif layer_name == 'Add':
        return l.AddLayer(layer, layer_name)

    else:
        print('[evaluate_layer()] layer: ', layer.name, layer_name, ' is not supported')
        return l.Layer(layer, layer_name)