import numpy as np
import h5py
import argparse
import pprint as pp

import tensorflow as tf
from sklearn.metrics import accuracy_score

from utils import loadDataH5, plot_loss

import matplotlib.pyplot as plt
plt.style.use("ggplot")


'''
The Learner class represents an ensemble member
'''
class Learner(object):

    def __init__(self, id, trainX, trainY, valX, valY, epochs, batch_size):
        self.name = 'learner' + str(id)
        
        self.inputShape = (128, 128, 3)
        self.classes = 17

        self.trainX = trainX
        self.trainY = trainY
        self.valX = valX
        self.valY = valY

        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size

        self.model = self.build()
        self.train_generator = self.data_augmentation()
        self.predictions = None
        self.score = None

        self.best_weights = self.name + "_weights.hdf5"

    def run(self):
        '''
        Main driver for Learner object
        '''
        self.train()
        self.load_weights()
        self.predict()

    def build(self):
        '''
        Dynamic CNN builder
        '''

        print('\nBuilding and compiling model:\n')

        '''
        Define feature maps per convBlock
        '''
        filters_sizes = {
            0:  64,
            1: 128,
            2: 256,
            3: 512
        }

        '''
        Define FC layer sizes
        '''
        dense_layers = {
            0: [1024, 512],
            1: [1024]
        }

        # Instantiate a Sequential model
        model = tf.keras.models.Sequential()

        # Add an InputLayer to allow for dynamic neural assembly 
        model.add(tf.keras.layers.InputLayer(input_shape=self.inputShape))
        
        for convBlock in range(np.random.choice([3, 4])):
            filters = filters_sizes[convBlock]
            conv = np.random.choice([3, 5])
            for _ in range(np.random.choice([1, 2, 3])):
                model.add(tf.keras.layers.Conv2D(filters, (conv, conv), padding='same', activation='relu'))
            
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())

        for size in dense_layers[np.random.choice([0, 1])]:
            model.add(tf.keras.layers.Dense(size, activation='relu'))
        
        model.add(tf.keras.layers.Dense(self.classes, activation='softmax'))

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
    
    def train(self):
        '''
        Training function. Define checkpoint callback, print best result, plot graph.
        '''

        print('\nTraining ensemble learner:', self.name)

        numTrainingSamples = self.trainX.shape[0]
        numValidationSamples = self.valX.shape[0]

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.best_weights, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)

        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=numTrainingSamples // self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_data=(self.valX, self.valY),
            validation_steps=numValidationSamples // self.BATCH_SIZE,
            callbacks=[checkpoint],
            verbose=1)
        
        print('\nMax validation accuracy:', max(history.history["val_accuracy"]))
        self.plot_loss(history, self.EPOCHS, 'Ensemble ' + self.name)
    
    def load_weights(self):
        '''
        Load the best checkpointed weights.
        '''
        print('\nLoading weights for lowest validation loss.')
        self.model.load_weights(self.best_weights)
        print('Done')
    
    def predict(self):
        '''
        Get predictions and score for validation set.
        '''
        print('\nPredicting validation set classes.')
        self.score = self.model.evaluate(self.valX, self.valY, verbose=0)
        print('Validation set score:', self.score)
        self.predictions = self.model.predict(self.valX, batch_size=self.BATCH_SIZE)
        print('Done')
    
    def data_augmentation(self):
        '''
        Create distinct ImageDataGenerator for each ensemble learner.
        '''
        data_gen_args = dict(
                rotation_range=np.random.randint(20, 75),
                zoom_range=np.random.uniform(0.1, 0.5),
                shear_range=np.random.uniform(0.1, 0.5),
                width_shift_range=np.random.uniform(0.1, 0.5), 
                height_shift_range=np.random.uniform(0.1, 0.5),
                horizontal_flip=True
            )

        train_gen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
        train_generator = train_gen.flow(self.trainX, self.trainY, batch_size=self.BATCH_SIZE)

        print('\nData augmentation with the following parameters:')
        pp.pprint(data_gen_args)

        return train_generator

    def plot_loss(self, history, epochs, name):
        print('\n\n')
        plt.figure(figsize=(12,8))
        plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
        plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, epochs), history.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy - {}".format(name))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.show()

'''
Ensemble class creating list of ensemble members
'''
class Ensemble(object):

    def __init__(self, n, epochs, batch_size):
        
        self.n = n
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size

        self.trainX, self.trainY, self.valX, self.valY = loadDataH5()

        self.ensemble = []
        self.ensemble_probabilities = 0
        self.ensemble_accuracy = None
    
    def run(self):
        '''
        Loop to create and train n Learners. Store learner predictions and score in ensemble list.
        '''
        for i in range(self.n):
            print("##################################################")
            print('\nLearner ' + str(i))
            learner = Learner(i, self.trainX, self.trainY, self.valX, self.valY, self.EPOCHS, self.BATCH_SIZE)
            learner.run()

            self.ensemble.append((learner.predictions, learner.score))
            self.ensemble_probabilities += learner.predictions

        '''
        Get averaged maximum prediction from accumulated ensembles predictions
        '''
        ensemble_predictions = (self.ensemble_probabilities/self.n).argmax(axis=1)
        self.ensemble_accuracy = accuracy_score(self.valY, ensemble_predictions)

        print('\n##################################################"')
        print('\nFinal ensemble accuracy:', self.ensemble_accuracy)

    def plot_ensemble_stats(self):
        '''
        Plot Ensemble stats
        '''
        accuracies = [ learner[1][1] for learner in self.ensemble]
        accuracies.append(self.ensemble_accuracy)

        xticks = [ str(i) for i in range(self.n) ]
        xticks.append('Ensemble Avg')

        plt.figure(figsize=(12,8))
        plt.bar(np.arange(0, len(accuracies)), accuracies, label="val_acc")
        plt.title("Validation Accuracy - Learners and Ensemble")
        plt.xticks(np.arange(0, len(accuracies)), xticks)
        plt.xlabel("Learner")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()


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

    parser.add_argument('--N', action='store', dest='n',
                        help='Num ensemble members', type=int)

    args = parser.parse_args()

    '''
    Read arguments
    '''
    n = args.n
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    ensemble = Ensemble(n, EPOCHS, BATCH_SIZE)
    ensemble.run()

