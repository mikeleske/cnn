import numpy as np
import h5py
import argparse

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

    '''
    Build model and train
    '''
    model = build_model(width, height, depth, classes, convBlocks)
    history = model.fit(trainX, trainY, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(valX, valY))

    print('\n\nMax validation accuracy:', max(history.history["val_accuracy"]))
    plot_loss(history, EPOCHS, 'Simple CNN Model - ' + str(convBlocks))