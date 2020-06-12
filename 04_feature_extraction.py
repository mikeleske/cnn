import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import np_utils

import argparse

from utils import loadDataH5

from sklearn import metrics
from sklearn import decomposition


pretrained_networks = ['VGG16', 'VGG19', 'InceptionV3', 'DenseNet121', 'DenseNet201',
    'ResNet152V2', 'MobileNet', 'MobileNetV2'
]

classifiers = ['LogisticRegression', 'SVC', 'SGDClassifier', 'RandomForestClassifier']


'''
Download pre-trained network with imagenet weights
'''
def download_pretrained_network(arch, input_shape):
    nn = None

    if arch == 'VGG16':
        nn = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif arch == 'VGG19':
        nn = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    elif arch == 'InceptionV3':
        nn = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    elif arch == 'DenseNet121':
        nn = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    elif arch == 'DenseNet201':
        nn = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)
    elif arch == 'ResNet152V2':
        nn = tf.keras.applications.ResNet152V2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif arch == 'MobileNet':
        nn = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    elif arch == 'MobileNetV2':
        nn = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    return nn

'''
Create secondary classifier
'''
def get_clf(clf):
    model = None

    if clf == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
    elif clf == 'SVC':
        from sklearn.svm import SVC
        model = SVC()
    elif clf == 'SGDClassifier':
        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier()
    elif clf == 'RandomForestClassifier':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(500)

    return model


'''
Extract feature from last CNN convolutional layer, then perform PCA.
'''
def get_nn_out(nn, trainX, valX, pca=None):
    featuresTrain = nn.predict(trainX)
    featuresTrain = featuresTrain.reshape(trainX.shape[0], -1)

    featuresVal = nn.predict(valX)
    featuresVal = featuresVal.reshape(valX.shape[0], -1)

    if pca:
        pca = decomposition.PCA(n_components=pca)
        pca.fit(featuresTrain)
        featuresTrain = pca.transform(featuresTrain)
        featuresVal = pca.transform(featuresVal)

    return featuresTrain, featuresVal

'''
Train secondary machine learning classifiers with extracted features
'''
def fit_and_evaluate(model, featuresTrain, trainY, featuresVal, valY, pca=None):
    model.fit(featuresTrain, trainY)

    print('\nClassifier:', type(model))
    print('PCA:', pca)
    
    # evaluate the model
    results_train = model.predict(featuresTrain)
    print('  Training accuracy:', metrics.accuracy_score(results_train, trainY))

    results_val = model.predict(featuresVal)
    print('  Validation accuracy:', metrics.accuracy_score(results_val, valY))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--NET', action='store', dest='net',
                        help='Pre-trained network', type=str)

    parser.add_argument('--CLF', action='store', dest='clf',
                        help='Secondary classifier', type=str)

    parser.add_argument('--PCA', action='store', dest='pca',
                        help='PCA', 
                        default=0, type=int)

    args = parser.parse_args()

    '''
    Read arguments
    '''
    net = args.net
    clf = args.clf
    pca = args.pca

    assert net in pretrained_networks, 'Requested network is not in pretrained network list.'
    assert clf in classifiers, 'Requested classifier is not in classifiers list.'

    '''
    Load the data
    '''
    trainX, trainY, valX, valY = loadDataH5()
    input_shape = (trainX.shape[1], trainX.shape[2], trainX.shape[-1])

    '''
    Download network and extract features
    '''
    nn = download_pretrained_network(net, input_shape)
    featuresTrain, featuresVal = get_nn_out(nn, trainX, valX, pca)

    '''
    Train secondary classifier with extracted features
    '''
    model = get_clf(clf)
    fit_and_evaluate(model, featuresTrain, trainY, featuresVal, valY, pca)