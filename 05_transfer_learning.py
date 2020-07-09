import argparse

from utils import loadDataH5
from Tuner import Tuner


pretrained_networks = ['VGG16', 'VGG19', 'InceptionV3', 'DenseNet121', 'DenseNet201',
    'ResNet152V2', 'MobileNet', 'MobileNetV2'
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--NET', action='store', dest='net',
                        help='Pre-trained network', type=str)

    parser.add_argument('--EPOCHS', action='store', dest='epochs',
                        help='Training epochs', 
                        default=50, type=int)

    parser.add_argument('--BATCH_SIZE', action='store', dest='batch_size',
                        help='Batch size',
                        default=32, type=int)

    parser.add_argument('--CLASSES', action='store', dest='classes',
                        help='Num classes', type=int)

    args = parser.parse_args()

    '''
    Load the data
    '''
    trainX, trainY, valX, valY = loadDataH5()
    
    '''
    Read arguments
    '''
    NET = args.net
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    CLASSES = args.classes

    assert NET in pretrained_networks, 'Requested network is not in pretrained network list.'

    '''
    Create and train a tuner
    '''
    model = Tuner(NET, trainX, trainY, valX, valY, CLASSES, EPOCHS, BATCH_SIZE)
    model.run()

