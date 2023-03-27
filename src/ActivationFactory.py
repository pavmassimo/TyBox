from .Activations import *


def create_activation(activation):
    print("create_activation")
    print(activation)

    if activation == 'linear':
        return Linear()

    elif activation == 'relu':
        return Relu()

    elif activation == 'softmax':
        return Softmax()

    elif activation == 'sigmoid':
        return Sigmoid()

    else:
        print('activation: ', activation, ' is not supported, using linear instead')
        return Linear()
