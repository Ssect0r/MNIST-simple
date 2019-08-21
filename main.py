#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import random

from tensorflow import keras

# Suppress some tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='relu'),
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


def display_samples(model, training_data):
    samples_num = int(input('How many samples would you like to see? '))
    for i in range(samples_num):
        rand_index = random.randint(0, len(training_data[0]))
        prediction = model.predict(training_data[0][rand_index:rand_index+1])
        for value, percentage in enumerate(np.round(prediction, decimals=3)[0]):
            print('{} has {:.2f}%% confidence'.format(value, 100 * percentage))
        print('Correct value:', training_data[1][rand_index].argmax())
        plt.imshow(training_data[0][rand_index])
        plt.show()


def main():
    adam_optimizer = keras.optimizers.Adam(lr=0.001)
    # Build and compile the model
    model = build_model()
    model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    training_data, test_data = prepare_data()
    use_saved_model = input('Do you want to use a saved model? [y/N] ').lower() == 'y'
    if not use_saved_model:
        print('Starting the learning process...'.format(model))
        model.fit(training_data[0], training_data[1], batch_size=32, epochs=5)
    else:
        print('Choose a file: ')
        for base, dirs, files in os.walk('./saved_models'):
            for file in files:
                if file.endswith('.index'):
                    name = file[:-6]
                    print('\t', name)
        chosen_file = input('\nModel name: ')
        if not os.path.exists(os.path.join(base, chosen_file + '.index')) or not chosen_file:
            print('ERROR: Model not found')
            sys.exit(1)
        model.load_weights('./saved_models/{}'.format(chosen_file))
    # Test the trained model on unknown data
    print('Running the evaluation test...')
    _, accuracy = model.evaluate(test_data[0], test_data[1])
    if not use_saved_model and input('Would you like to save this model? [y/N] ').lower() == 'y':
        filename = input('File name: ')
        model.save_weights('saved_models/{}'.format(filename), overwrite=True)
        print('Saved {}'.format(filename))
    model.summary()
    print('Accuracy: {:.2f}%'.format(accuracy * 100))
    display_samples(model, training_data)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nProgram interrupted')
    else:
        print('Program finished successfully')
