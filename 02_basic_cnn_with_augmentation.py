import numpy as np
import h5py
import argparse

import tensorflow as tf

from utils import loadDataH5, plot_loss, convert_conv_blocks, build_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--EPOCHS', action='store', dest='epochs',
                        help='Training epochs', 
                        default=50, type=int)

    parser.add_argument('--BATCH_SIZE', action='store', dest='batch_size',
                        help='Batch size',
                        default=32, type=int)

    parser.add_argument('--CLASSES', action='store', dest='classes',
                        help='Num classes', type=int)

    parser.add_argument('--CONVBLOCKS', action='store', dest='conv_blocks',
                        help='Conv block architecture', type=str)

    args = parser.parse_args()

    '''
    Read arguments
    '''
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    classes = args.classes
    convBlocks = convert_conv_blocks(args.conv_blocks)

    '''
    Load the data
    '''
    trainX, trainY, valX, valY = loadDataH5()
    width = trainX.shape[1]
    height = trainX.shape[2]
    depth = trainX.shape[-1]
    numTrainingSamples = trainX.shape[0]
    numValidationSamples = valX.shape[0]

    '''
    Define parameters for image data augmentation
    '''
    data_gen_args = dict(
            rotation_range=30,
            zoom_range=0.2,
            shear_range=0.2,
            width_shift_range=0.2, 
            height_shift_range=0.2,
            horizontal_flip=True,
            #vertical_flip=True,
        )

    '''
    Initialize ImageDataGenerator object
    '''
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    train_generator = train_gen.flow(trainX, trainY, batch_size=BATCH_SIZE)

    '''
    Build model and train
    '''
    model = build_model(width, height, depth, classes, convBlocks)
    history = model.fit(
        train_generator,
        steps_per_epoch=numTrainingSamples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(valX, valY),
        validation_steps=numValidationSamples // BATCH_SIZE)

    print('\n\nMax validation accuracy:', max(history.history["val_accuracy"]))
    plot_loss(history, EPOCHS, 'Simple CNN Model - ' + str(convBlocks) + '\n with data augmentation')