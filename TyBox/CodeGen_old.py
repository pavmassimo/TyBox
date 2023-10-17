import os
import subprocess

def extract_Mc_weights_as_string_unrolled_matrix(tf_model):
    result = []
    for layer in tf_model.layers:
        if len(layer.weights) > 0:
            hiddel_layer_weights = '{'
            for i in range(layer.weights[0].numpy().shape[0]):
                # hiddel_layer_weights += '{'
                for ii in range(len(layer.weights[0].numpy()[i])):
                    weight = layer.weights[0].numpy()[i][ii]
                    hiddel_layer_weights += str(weight)
                    if ii < len(layer.weights[0].numpy()[i]) - 1:
                        hiddel_layer_weights += ', '
                # hiddel_layer_weights += '}'
                if i < (layer.weights[0].numpy().shape[0] - 1):
                    hiddel_layer_weights += ',\n'
            hiddel_layer_weights += '}'

            hidden_layer_biases = '{'
            for i in range(layer.weights[1].numpy().shape[0]):
                hidden_layer_biases += str(layer.weights[1].numpy()[i])
                if i < (layer.weights[1].numpy().shape[0] - 1):
                    hidden_layer_biases += ', '
            hidden_layer_biases += '}'

            result.append((hiddel_layer_weights, hidden_layer_biases))
    return result

# input: dense model with inputs, hidden layer and output layer
# output: string with struct definition of the same model in C with weights (header file)
def generate_Mc_manual_C(M_c, len_buff):
    inputs = M_c.layers[0].input.shape[1]
    # hidden_nodes = M_c.layers[1].input.shape[1]
    outputs = M_c.output.shape[1]
    activations_function = []
    weights_strings = extract_Mc_weights_as_string_unrolled_matrix(M_c)
    layers = []
    for l in M_c.layers:
        activations_function.append(str(l.activation).split(" ")[1])
        if 'dense' in l.name:
            layers.append(l.input.shape[1])
    layers.append(M_c.output.shape[1])
    layer_sizes_list = '{'
    for layer_index in range(len(layers)):
        layer_sizes_list += str(layers[layer_index])
        if layer_index < len(layers) - 1:
            layer_sizes_list += ', '
    layer_sizes_list += '}'

    string = ''
    string += gen_header(inputs, len_buff, len(layers)-1)
    string += gen_activations()

    string += gen_network_struct(layer_sizes_list)

    string += f"float activation_0[{layers[0]}];\n"

    for index in range(len(layers) - 1):
        string += "float activation_{index_1}[{size}];\n\
float bias_layer_{index}[{size}] = {biases};\n\
float weights_layer_{index}[{weight_size}] = {weights};\n\
Activation_function activation_function_{index} = new {activation_function}();\n" \
            .format(index=index, index_1=index+1, size=layers[index+1],
                    weights=weights_strings[index][0], biases=weights_strings[index][1],
                    weight_size=layers[index + 1] * layers[index],
                    activation_function=activations_function[index].capitalize())
    string += gen_init_network(len(layers))

    string += gen_forward_pass()
    string += gen_get_label()
    string += gen_update_weights()
    string += gen_backpropagate()
    string += gen_train()
    string += gen_push()
    string += gen_evaluate()

    return string

def gen_header(feature_size, buffer_size, n_layers):
    res = f"#include <math.h>\n\
#include <assert.h>\n\
#include <stdio.h>\n\
\n\
#define FEATURE_SIZE {feature_size}\n\
#define BUFFER_SIZE {buffer_size}\n\
#define N_LAYERS {n_layers}\n\
"
    return res

def gen_activations():
    res = "class Activation_function{\n\
  public:\n\
    virtual void activate(float* vector, int len_vector) {\n\
    };\n\
    virtual void derivative(float* vector, int len_vector, float* der_vector) {\n\
    };\n\
    \n\
    virtual void calculate_output_deltas(float* vector, int len_vector, int* target, float* delta_vector) {\n\
        derivative(vector, len_vector, delta_vector);\n\
        for(int i = 0; i < len_vector; i++){\n\
            float errorOutput = float(target[i]) - vector[i];\n\
            delta_vector[i] *= errorOutput;\n\
        }\n\
    }\n\
    \n\
    virtual void calculate_hidden_deltas(float* vector, int len_vector, float* weights_vector, float* successive_layer_deltas,\n\
                                 int len_successive_delta, float* delta_vector) {\n\
        derivative(vector, len_vector, delta_vector);\n\
        for(int i = 0; i < len_vector; i++){\n\
            float sum_of_succ_layer_deltas = 0;\n\
            for (int j = 0; j < len_successive_delta; j++) {\n\
                sum_of_succ_layer_deltas += (successive_layer_deltas[j] * weights_vector[i*len_successive_delta + j]);\n\
            }\n\
            delta_vector[i] *= sum_of_succ_layer_deltas;\n\
        }\n\
    }\n\
\n\
};\n\
\n\
\n\
class Sigmoid: public Activation_function{\n\
  void activate(float * vector, int len_vector) override{\n\
    for (int i=0; i < len_vector; i++){\n\
      vector[i] = 1 / (1 + exp(-vector[i]));\n\
    }\n\
  }\n\
\n\
  void derivative(float* vector, int len_vector, float* der_vector) override{\n\
      for (int i=0; i < len_vector; i++){\n\
          der_vector[i] = vector[i] * (1 - vector[i]);\n\
      }\n\
  }\n\
\n\
};\n\
\n\
class Relu: public Activation_function{\n\
  void activate(float * vector, int len_vector) override{\n\
      for (int i=0; i < len_vector; i++){\n\
          vector[i] = vector[i] > 0 ? vector[i] : 0;\n\
      }\n\
  }\n\
\n\
  void derivative(float* vector, int len_vector, float* der_vector) override{\n\
      for (int i=0; i < len_vector; i++){\n\
          der_vector[i] = vector[i] > 0 ? 1 : 0;\n\
      }\n\
  }\n\
};\n\
\n\
class Linear: public Activation_function{\n\
  void activate(float * vector, int len_vector) override{\n\
  }\n\
\n\
  void derivative(float* vector, int len_vector, float* der_vector) override{\n\
      for (int i=0; i < len_vector; i++){\n\
          der_vector[i] = 1;\n\
      }\n\
  }\n\
};\n\
\n\
class Softmax: public Activation_function{\n\
    void activate(float * vector, int len_vector) override{\n\
    \n\
        int i;\n\
        float m, sum, constant;\n\
    \n\
        m = -INFINITY;\n\
        for (i = 0; i < len_vector; ++i) {\n\
            if (m < vector[i]) {\n\
                m = vector[i];\n\
            }\n\
        }\n\
    \n\
        sum = 0.0;\n\
        for (i = 0; i < len_vector; ++i) {\n\
            sum += exp(vector[i] - m);\n\
        }\n\
    \n\
        constant = m + log(sum);\n\
        for (i = 0; i < len_vector; ++i) {\n\
            vector[i] = exp(vector[i] - constant);\n\
        }\n\
    \n\
    }\n\
    \n\
    void derivative(float* vector, int len_vector, float* der_vector) override{\n\
        float * res = der_vector;\n\
    \n\
        for(int i=0;i<len_vector;i++){\n\
            for(int j=0;j<len_vector;j++){\n\
                res[i* len_vector + j]+=vector[i]*vector[j];\n\
            }\n\
        }\n\
    \n\
        for (int i = 0; i <len_vector; i++) {\n\
            res[i*len_vector +i] += vector[i];\n\
        }\n\
    \n\
    }\n\
    \n\
    void calculate_output_deltas(float* vector, int len_vector, int* target, float* delta_vector) override {\n\
        float der_vector [len_vector*len_vector];\n\
        for (int i =0; i < len_vector*len_vector; i++){\n\
            der_vector[i] = 0;\n\
        }\n\
        derivative(vector, len_vector, der_vector);\n\
        float errorOutput;\n\
    \n\
    \n\
        for(int i = 0; i < len_vector; i++){\n\
            delta_vector[i] = 0;\n\
            for(int j = 0; j < len_vector; j++){\n\
                errorOutput = float(target[j]) - vector[j];\n\
                delta_vector[i] += der_vector[i*len_vector + j] * errorOutput;\n\
            }\n\
        }\n\
    }\n\
    \n\
    void calculate_hidden_deltas(float* vector, int len_vector, float* weights_vector, float* successive_layer_deltas,\n\
                                 int len_successive_delta, float* delta_vector) override {\n\
        //not implemented yet!\n\
    }\n\
};"
    return res


def gen_network_struct(layer_sizes_list):
    res = "typedef struct t_dense_network {{ \n\
        float * bias_list[N_LAYERS];\n\
        float * weight_list[N_LAYERS];\n\
        float * activations_list[N_LAYERS+1];\n\
        Activation_function* activation_functions[N_LAYERS];\n\
        \n\
        const int number_of_activations = N_LAYERS+1;\n\
        const int act_sizes[N_LAYERS+1] = {};\n\
        const int numOutputs = act_sizes[number_of_activations - 1];\n\
    \n\
        float mbp_buffer[BUFFER_SIZE][FEATURE_SIZE + 1]; //last number in the list is the label\n\
        int buf_index;\n\
        bool is_full;\n\
    \n\
    \n\
    }}t_dense_network;\n"\
    .format(layer_sizes_list)
    return res

def gen_init_network(number_of_layers):

    res = "void init_network(t_dense_network &dense_network){\n\
    dense_network.activations_list[0] = activation_0;\n"

    for index in range(number_of_layers - 1):
        res += "    dense_network.activations_list[{index_one}] = activation_{index_one};\n\
    dense_network.bias_list[{index}] = bias_layer_{index};\n\
    dense_network.weight_list[{index}] = weights_layer_{index};\n\
    dense_network.activation_functions[{index}] = activation_function_{index}\n" \
            .format(index=index, index_one=(index + 1))

    res += '    dense_network.buf_index = 0;\n\
    dense_network.is_full = false;\n\
    }\n'
    return res




def gen_forward_pass():
    res = "void forward_pass(t_dense_network & d_network){\n\
    \n\
    for (int act_index = 1; act_index < d_network.number_of_activations; act_index++){\n\
    \n\
        for (int node_index = 0; node_index < d_network.act_sizes[act_index]; node_index++){\n\
            float act = 0;\n\
            for (int input_node_index = 0; input_node_index < d_network.act_sizes[act_index - 1]; input_node_index++) {\n\
                float current_input = d_network.activations_list[act_index - 1][input_node_index];\n\
                float current_weight = d_network.weight_list[act_index - 1][input_node_index * d_network.act_sizes[act_index] + node_index];\n\
                act += current_input * current_weight;\n\
            }\n\
            float current_bias = d_network.bias_list[act_index - 1][node_index];\n\
            act = act + current_bias;\n\
            d_network.activations_list[act_index][node_index] = act;\n\
        }\n\
        d_network.activation_functions[act_index - 1]->activate(d_network.activations_list[act_index], d_network.act_sizes[act_index]);\n\
    }\n\
}\n"
    return res


def gen_get_label():
    res = "int get_label(t_dense_network &d_network){\n\
  float max_activation = 0;\n\
  int label = 0;\n\
  for(int output_index=0; output_index < d_network.numOutputs; output_index++){\n\
    if(d_network.activations_list[d_network.number_of_activations - 1][output_index] > max_activation){\n\
      max_activation = d_network.activations_list[d_network.number_of_activations - 1][output_index];\n\
      label = output_index;\n\
    }\n\
  }\n\
  return label;\n\
}\n"
    return res

def gen_update_weights():
    res = "void update_weights_and_biases_of_layer(int act_index, float *layer_deltas, float lr, t_dense_network &d_network){\n\
  int layer_size = d_network.act_sizes[act_index];\n\
  int previous_layer_size = d_network.act_sizes[act_index - 1];\n\
\n\
  for(int node_index = 0; node_index < layer_size; node_index++){\n\
    for(int previous_layer_node_index = 0; previous_layer_node_index < previous_layer_size; previous_layer_node_index++){\n\
      float weight_update = lr * layer_deltas[node_index] * d_network.activations_list[act_index - 1][previous_layer_node_index];\n\
      float weight = d_network.weight_list[act_index - 1][previous_layer_node_index * d_network.act_sizes[act_index] + node_index];\n\
      float new_weight = weight + weight_update;\n\
        d_network.weight_list[act_index - 1][previous_layer_node_index * d_network.act_sizes[act_index] + node_index] = new_weight;\n\
    }\n\
    \n\
      d_network.bias_list[act_index - 1][node_index] += lr * layer_deltas[node_index];\n\
  }\n\
}\n\
"
    return res

def gen_backpropagate():
    res = "void backpropagate(int *targets, float lr, t_dense_network &d_network){\n\
\n\
  float deltaOutput[d_network.numOutputs];\n\
\n\
\n\
  d_network.activation_functions[d_network.number_of_activations - 2]->calculate_output_deltas(d_network.activations_list[d_network.number_of_activations - 1],\n\
                                                              d_network.numOutputs, targets, deltaOutput);\n\
\n\
  update_weights_and_biases_of_layer(d_network.number_of_activations - 1, deltaOutput, lr, d_network);\n\
  float *successive_layer_deltas = deltaOutput;\n\
\n\
\n\
  // for all layers except output (from last to first)\n\
  for(int act_index = d_network.number_of_activations - 3; act_index > 0; act_index--){\n\
    \n\
    // calculate deltas\n\
    float deltas[d_network.act_sizes[act_index]];\n\
    d_network.activation_functions[act_index]->calculate_hidden_deltas(d_network.activations_list[act_index],\n\
                                                                       d_network.act_sizes[act_index],\n\
                                                                       d_network.weight_list[act_index],\n\
                                                                       successive_layer_deltas,\n\
                                                                       d_network.act_sizes[act_index + 1],\n\
                                                                       deltas);\n\
    \n\
    // update layer\n\
    update_weights_and_biases_of_layer(act_index, deltas, lr, d_network);\n\
    \n\
    // set succ layer deltas\n\
    successive_layer_deltas = deltas;\n\
  }\n\
}\n\
"
    return res

def gen_train():
    res = "void train(t_dense_network & d_network, float lr){\n\
    \n\
        for(int i=0; i<max(d_network.buf_index, BUFFER_SIZE * d_network.is_full); i++){\n\
    \n\
          // build categorical target\n\
          int targets[d_network.numOutputs];\n\
          for(int ii=0; ii<d_network.numOutputs; ii++){\n\
            targets[ii] = 0;\n\
          }\n\
          targets[int(d_network.mbp_buffer[i][FEATURE_SIZE])] = 1;\n\
    \n\
          //load features into the network\n\
          for(int j=0; j<FEATURE_SIZE; j++){\n\
              d_network.activations_list[0][j] = d_network.mbp_buffer[i][j];\n\
          }\n\
          //train\n\
          for(int rep=0; rep<1; rep++){\n\
            forward_pass(d_network);\n\
          }\n\
          for(int rep=0; rep<1; rep++){\n\
            backpropagate(targets, lr, d_network);\n\
          }\n\
        }\n\
}\n"
    return res


def gen_push():
    res = "void push_datum_into_buf(t_dense_network & d_network, float * output, int label){//TfLiteTensor* output, int label){\n\
    \n\
        // put features calculated by convolutional block into the input to the dense block and save them into the buffer\n\
        for (int input_node_index=0; input_node_index < d_network.act_sizes[0]; input_node_index++){\n\
          if(d_network.buf_index < BUFFER_SIZE){\n\
              //d_network.mbp_buffer[d_network.buf_index][input_node_index] = float(output->data.f[input_node_index]);\n\
              d_network.mbp_buffer[d_network.buf_index][input_node_index] = output[input_node_index];\n\
          }\n\
          else{\n\
              //d_network.mbp_buffer[0][input_node_index] = float(output->data.f[input_node_index]);\n\
              d_network.mbp_buffer[0][input_node_index] = output[input_node_index];\n\
          }\n\
        }\n\
    \n\
        //save the label\n\
        if(d_network.buf_index < BUFFER_SIZE){\n\
            d_network.mbp_buffer[d_network.buf_index][FEATURE_SIZE] = label;\n\
          d_network.buf_index++;\n\
        }\n\
        else{\n\
            d_network.is_full = true;\n\
            d_network.mbp_buffer[0][FEATURE_SIZE] = label;\n\
            d_network.buf_index = 1;\n\
        }\n\
}\n\
"
    return res

def gen_evaluate():
    res = "bool evaluate(t_dense_network & d_network, float * output, int label){// TfLiteTensor* output, int label){\n\
        //load into the network\n\
        for(int input_node_index=0; input_node_index<FEATURE_SIZE; input_node_index++){\n\
            //d_network.activations_list[0][input_node_index] = float(output->data.f[input_node_index]);\n\
            d_network.activations_list[0][input_node_index] = float(output[input_node_index]);\n\
        }\n\
        //calculate output\n\
        forward_pass(d_network);\n\
        int l = get_label(d_network);\n\
        return label == l;\n\
}\n"
    return res

def create_Mf_lite_C(Mf_lite):
    model_name = 'Mf_tflite'
    Mf_header = "/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.\n\
  Licensed under the Apache License, Version 2.0 (the \"License\");\n\
  you may not use this file except in compliance with the License.\n\
  You may obtain a copy of the License at\n\
      http://www.apache.org/licenses/LICENSE-2.0\n\
  Unless required by applicable law or agreed to in writing, software\n\
  distributed under the License is distributed on an \"AS IS\" BASIS,\n\
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n\
  See the License for the specific language governing permissions and\n\
  limitations under the License.\n\
  ==============================================================================*/\n\
  #ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MODEL_H_\n\
  #define TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MODEL_H_\n\
  extern const unsigned char {model_name}[];\n\
  extern const int {model_name}_len;\n\
  #endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MODEL_H_".format(model_name=model_name)

    with open('Mf_lite.tflite', 'wb') as file:
        file.write(Mf_lite)
    os.system("!xxd -i Mf_lite.tflite > Mf_lite.cc")  # c:\cygwin64\bin\bash.exe --login -c xxd -i Mf_lite.tflite > Mf_lite.cc
    sub_proc_result = subprocess.run(["xxd", "-i", "Mf_lite.tflite", "Mf_lite.cc"], capture_output=True)
    assert sub_proc_result.returncode == 0
    with open('Mf_lite.cc', 'r') as file:
        Mf_cc = file.read()
    assert len(Mf_cc) > 0
    os.remove('Mf_lite.tflite')
    os.remove('Mf_lite.cc')
    return Mf_header, Mf_cc
