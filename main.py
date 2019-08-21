#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging

from tensorflow import keras

# Suppress some tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model


def prepare_data():
    # Training data contains 2 arrays - images and digits
    (xtrain, ytrain), (xtest, ytest) = keras.datasets.mnist.load_data()
    # Convert data into 10 digits categorially
    ytrain = keras.utils.to_categorical(ytrain, num_classes=10)
    ytest = keras.utils.to_categorical(ytest, num_classes=10)
    return (xtrain / 255, ytrain), (xtest / 255, ytest)


def main():
    adam_optimizer = keras.optimizers.Adam(lr=0.001)
    model = build_model()
    training_data, test_data = prepare_data()
    model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    logging.info('Starting the learning process with {}'.format(model))
    model.fit(training_data[0], training_data[1], batch_size=16, epochs=3)
    # Test the trained model on unknown data
    logging.info('Evaluation test running...')
    model.evaluate(test_data[0], test_data[1])


if __name__ == '__main__':
    main()
