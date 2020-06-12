import numpy as np
import h5py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
plt.style.use("ggplot")

'''
Function to load the data from disk
'''
def loadDataH5():
    with h5py.File('data1.h5','r') as hf:
        trainX = np.array(hf.get('trainX'))
        trainY = np.array(hf.get('trainY'))
        valX = np.array(hf.get('valX'))
        valY = np.array(hf.get('valY'))

        print('Training data:', trainX.shape, trainY.shape)
        print('Test data    :', valX.shape, valY.shape)

    return trainX, trainY, valX, valY


'''
Function to build and return a basic CNN model
'''
def build_model(width, height, depth, classes, convBlocks=[]):
    '''
    Function to build simple CNNs in a dynamical way.
    convBlock provides nested lists of feature maps per Conv2D Layer
    '''

    # Define the inputShape needed for the InputLayer
    inputShape = (width, height, depth)

    # Instantiate a Sequential model
    model = tf.keras.models.Sequential()

    # Add an InputLayer to allow for dynamic neural assembly 
    model.add(tf.keras.layers.InputLayer(input_shape=inputShape))
    
    for convBlock in convBlocks:
        for convFilter in convBlock:
             model.add(tf.keras.layers.Conv2D(convFilter, (3, 3), padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dense(classes, activation='softmax'))

    # Define the optimizer as per assignment specification
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)

    # Compile the model
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    
    # Print the model summary
    model.summary()
        
    # return the constructed network architecture
    return model


'''
Function to plot training/validation loss/accuracy
'''
def plot_loss(history, epochs, configuration):
    print('\n\n')
    plt.figure(figsize=(12,8))
    plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy - {}".format(configuration))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


'''
Function to convert the CLI input
'''
def convert_conv_blocks(convBlocks):
    return [ list(map(int, x.split())) for x in convBlocks.split(',') ]