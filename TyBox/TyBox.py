import math

import tensorflow as tf
import numpy as np

from .Profiler import Profiler

from .CodeGen import *

from .PyBox import Model

class Error(Exception):
    pass


class ModelRequirementsError(Error):
    def __init__(self, message):
        self.message = message

# toolbox

# input: tf_model; requirement: tflite supported feature extraction block + flatten layer + dense block
# input: yield_representative_dataset
# return: model1, feature extraction block + flatten layer as tf model
# return: model2, dense block as tf model
def split_model(tf_model, yield_representative_dataset):
    # calculate index of flatten layer (the model is split at the output of the flatten layer)
    flatten_layers = list(filter(lambda layer: 'flatten' in layer.name, tf_model.layers))
    if len(flatten_layers) == 1:
        flatten_index = tf_model.layers.index(flatten_layers[0])

        layers1 = tf_model.layers[0:flatten_index + 1]
        layers2 = tf_model.layers[flatten_index + 1:]
        # print('debug', layers2[0].input.shape, layers2[0].output.shape)

        # first part of the model: feature extraction through convolutions
        model1 = tf.keras.Sequential()
        for i in layers1:
            model1.add(i)
        model1.compile()
        model1.build(tf_model.input.shape)


        # second part of the model: feature classification with dense feedforward network
        model2 = tf.keras.Sequential()
        # model2.add(tf.keras.layers.InputLayer(input_shape=model1.output.shape))
        for i in layers2:
            model2.add(i)
        model2.compile()

        # in1 = next(yield_representative_dataset())[0].numpy()
        in1 = np.random.rand(model1.input.shape[1], model1.input.shape[2], model1.input.shape[3])
        out1 = model1.predict(in1.reshape(1, model1.input.shape[1], model1.input.shape[2], model1.input.shape[3]))
        out2 = model2.predict(out1)
        # print(len(model1.layers), len(model2.layers))
        return model1, model2
    else:
        raise ModelRequirementsError('there is no flatten layer')


def extract_weights(tf_model):
    result = []
    for layer in tf_model.layers:
        weights = ''
        if len(layer.weights) > 0:
            weights += str(layer.weights[0].numpy().shape[0]) + ' '
            weights += str(layer.weights[0].numpy().shape[1]) + '\n'
            for i in layer.weights[0].numpy():
                for ii in range(len(i)):
                    weight = i[ii]
                    weights += str(weight) + ' '
                weights += '\n'
            weights += str(layer.weights[1].numpy().shape[0]) + '\n'
            for i in layer.weights[1].numpy():
                weights += str(i) + ' '
            weights += '\n'
            result.append(weights)
    return result


# input: model, tf model to convert
# input: yield_representative_dataset
def convert_to_tflite(model, yield_representative_dataset=None):
    if yield_representative_dataset:
        converter_model_lite = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_model_lite.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_model_lite.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # setting inference_input/output_type = tf.int8 removes the quantization / dequantization modules from the tflite model
        # so we will have to pass quantized inputs and will receive quantized values as outputs
        # not setting that, the model will have these modules and we can interface it with float values
        # converter.inference_input_type = tf.int8
        # converter.inference_output_type = tf.int8
        # converter_model_lite.representative_dataset = representative_dataset_mnist
        converter_model_lite.representative_dataset = yield_representative_dataset
        model_lite = converter_model_lite.convert()
    else:
        converter_Mf_lite_bare = tf.lite.TFLiteConverter.from_keras_model(model)
        model_lite = converter_Mf_lite_bare.convert()
    return model_lite


def extract_Mc_weights_as_list(tf_model):
    result = []
    for layer in tf_model.layers:
        if len(layer.weights) > 0:
            hiddel_layer_weights = []
            # layer = M_c.layers[0]
            for i in range(layer.weights[0].numpy().shape[0]):
                hiddel_layer_weights.append([])
                for ii in range(len(layer.weights[0].numpy()[i])):
                    weight = layer.weights[0].numpy()[i][ii]
                    hiddel_layer_weights[i].append(weight)
            hidden_layer_biases = []
            for i in layer.weights[1].numpy():
                hidden_layer_biases.append(i)
            result.append((hiddel_layer_weights, hidden_layer_biases))
    return result


# tf_model must be the classification head of the solution
def generate_implementation(tf_model):
    Mc_manual_python = create_python_model(tf_model)
    Mc_manual_header = generate_Mc_manual_C(tf_model)
    return Mc_manual_python, Mc_manual_header


def create_python_model(M_c):
    # create model
    layers = []
    #   layer.input.shape[1] for layer in M_c.layers
    # if len(M_c.layers) == 1:
    #     layers.append(M_c.layers[0].input.shape[1])
    #     layers.append(M_c.layers[0].output.shape[1])
    # else:
    for l in M_c.layers:
        if 'dense' in l.name:
            layers.append(l.input.shape[1])
    layers.append(M_c.layers[-1].output.shape[1])
    print('debug', layers)
    # print('layers', layers)
    Mc_manual_python = Model(layers)
    # transfer weights
    w = extract_weights(M_c)
    file_name_list = []
    for i in range(len(w)):
        file_name = "weights_{}".format(i)
        file_name_list.append(file_name)
        with open(file_name, 'w') as file:
            file.write(w[i])
    # print(file_name_list)
    # print(len(M_c.layers))
    Mc_manual_python.read_weights(file_name_list)
    for file_name in file_name_list:
        os.remove(file_name)
    return Mc_manual_python


# input: tf_model; requirement: tflite supported feature extraction block + flatten layer + dense block (3 layers: input, hidden, output)
# input: yield_representative_dataset
def create_on_device_learning_solution(tf_model, yield_representative_dataset):
    # M_f: Model_features, feature extraction block + flatten layer
    # M_c: Model_classification, dense block
    M_f, M_c = split_model(tf_model, yield_representative_dataset)
    Mf_lite = convert_to_tflite(M_f, yield_representative_dataset)
    Mc_lite = convert_to_tflite(M_c)
    Mc_manual_python = create_python_model(M_c)
    Mc_manual_header = generate_Mc_manual_C(M_c)
    Mf_header, Mf_cc = create_Mf_lite_C(Mf_lite)
    return Mf_lite, Mc_manual_python, M_c, Mc_manual_header, Mf_header, Mf_cc


# input: tf_model; requirement: tflite supported feature extraction block + flatten layer + dense block (3 layers: input, hidden, output)
# input: yield_representative_dataset
def create_python_learning_solution(tf_model, mem_available, precision):# yield_representative_dataset):
    # M_f: Model_features, feature extraction block + flatten layer
    # M_c: Model_classification, dense block
    try:
        M_f, M_c = split_model(tf_model) #, yield_representative_dataset)
        Mf_lite = convert_to_tflite(M_f)  # , yield_representative_dataset)
    except:
        print("model has no Flatten layer, assuming only dense model")
        M_c = tf_model
        Mf_lite = None

    input_dim = M_c.layers[0].input.shape[1] + 1

    len_buff = int(calculate_buf_dim(tf_model, mem_available, input_dim, precision))

    print(f"each datum requires {((input_dim) * precision / 8)} B")
    print(f"the buffer will be {len_buff} data long")

    Mc_manual_python = create_python_model(M_c, len_buff)
    return Mf_lite, Mc_manual_python

def calculate_buf_dim(tf_model, mem_available, input_dim, precision):
    prof = Profiler("new_model", tf_model)
    network_mem = prof.tot_mem_required
    print(f"{mem_available} B available")
    print(f"{network_mem} B dedicated to the network")

    buff_mem = mem_available - network_mem

    print(f"{buff_mem} B dedicated to the buffer")

    len_buff = buff_mem // (input_dim * precision / 8)

    return len_buff
