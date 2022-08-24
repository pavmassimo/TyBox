import TyBox.Activations as a

def create_activation(activation):
    print("create_a")
    print(activation)

    if activation == 'linear':
        return a.Linear()

    elif activation == 'relu':
        return a.Relu()

    elif activation == 'softmax':
        return a.Softmax()

    elif activation == 'sigmoid':
        return a.Sigmoid()

    else:
        print('activation: ', activation, ' is not supported, using linear instead')
        return a.Linear()