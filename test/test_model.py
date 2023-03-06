import os

import numpy as np
import pandas as pd
import pytest
from tensorflow import keras
import tensorflow as tf

from TyBox import split_model, create_python_model


def test_model():
    cwd = os.getcwd()

    filters = 8
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(4, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(keras.layers.MaxPool2D(2, 2))
    model.add(keras.layers.Conv2D(filters, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), metrics='accuracy')

    path = tf.keras.utils.get_file('mnist.npz', os.path.join(cwd, "resources/mnist.npz"))
    with np.load(path) as data:
        train_examples = np.array(data['x_train'], dtype='float32')
        train_labels = data['y_train']
        test_examples = np.array(data['x_test'], dtype='float32')
        test_labels = data['y_test']

    layer = tf.keras.layers.Normalization(axis=None)
    layer.adapt(train_examples)
    train_examples = np.array(layer(train_examples))
    test_examples = np.array(layer(test_examples))

    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    model.fit(train_dataset, epochs=1)
    model.evaluate(test_dataset)
    model.save("resources/test_model")
    return

def test_split():
    test_model = keras.models.load_model("resources/test_model")
    feature_extractor, classifier = split_model(test_model, None)
    tybox_classifier_simulator = create_python_model(classifier)
    test_input = np.reshape(np.array([1 for _ in range(200)]), (1, -1))
    print(classifier.predict(test_input))
    tybox_classifier_simulator.execute_forward_pass(np.array([1 for _ in range(200)]))





