#encoding=utf-8
# Copyright@Chenggang Wang
# wang2c9@mail.uc.edu
# All right reserved
#
# April 26, 2019

import os
import sys
import argparse
import logging

if '1' == os.getenv('useGpu'):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
from keras.layers import Dense, Dropout, LSTM, Flatten
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.utils import np_utils
import keras.backend as K

import pandas as pd
import numpy as np

sys.path.append('bin')
from prepareData import loadTrainAndTestData

modelDir = 'modelDir'
if not os.path.isdir(modelDir):
    os.makedirs(modelDir)
LOG = logging.getLogger('modelDir/lstm_results')


def generate_default_params():
    return {
            'optimizer': 'Adamax',
            'activation': 'tanh',
            'rc_act': 'hard_sigmoid',
            'dropout_rate1': 0.4,
            'dropout_rate2': 0.1,
            'dropout_rate3': 0.1,
            'dropout_rate4': 0.3,
            'dropout_rate5': 0.5,
            'batch_size': 130,
            'decay': 0.1,
            'epochs': 500,
            'data_dim': 350,
            'layer1': 210,
            'layer2': 190,
            'layer3': 190,
            'layer4': 190,
            'layer5': 70,
            'dense': 70,
            'dense_act': 'selu',
            'kernel_ini': 'glorot_normal'
            }


class LSTM_Model():
    def __init__(self, opts, params, name='lstm'):
        self.params = params
        self.input = opts.input
        self.verbose = opts.verbose
        self.plotModel = opts.plotModel
        self.name = name


    def create_model(self, NUM_CLASS):
        print ('Creating model...')
        layers = [LSTM(self.params['layer1'], activation=self.params['activation'], input_shape=(self.params['data_dim'], 1), return_sequences=True, recurrent_activation=self.params['rc_act'], kernel_initializer=self.params['kernel_ini']),
                  Dropout(rate=self.params['dropout_rate1']),

                  LSTM(self.params['layer2'], activation=self.params['activation'], return_sequences=True, recurrent_activation=self.params['rc_act'], kernel_initializer=self.params['kernel_ini']),
                  Dropout(rate=self.params['dropout_rate2']),

                  LSTM(self.params['layer3'], activation=self.params['activation'], return_sequences=True, recurrent_activation=self.params['rc_act'], kernel_initializer=self.params['kernel_ini']),
                  Dropout(rate=self.params['dropout_rate3']),

                  LSTM(self.params['layer4'], activation=self.params['activation'], return_sequences=True, recurrent_activation=self.params['rc_act'], kernel_initializer=self.params['kernel_ini']),
                  Dropout(rate=self.params['dropout_rate4']),

                  LSTM(self.params['layer5'], activation=self.params['activation'], return_sequences=True, recurrent_activation=self.params['rc_act'], kernel_initializer=self.params['kernel_ini']),
                  Dropout(rate=self.params['dropout_rate5']),

                  Flatten(),
                  Dense(self.params['dense'], activation=self.params['dense_act'], kernel_initializer=self.params['kernel_ini']),
                  Dense(NUM_CLASS, activation='softmax', kernel_initializer='glorot_normal')]

        model = Sequential(layers)

        print ('Compiling...')
        model.compile(loss='categorical_crossentropy',
                     optimizer=self.params['optimizer'],
                     metrics=['accuracy'])
        return model


    def train(self, X_train, y_train, NUM_CLASS):
        model = self.create_model(NUM_CLASS)
        if self.plotModel:
            picDir = os.path.join(modelDir, 'pic')
            if not os.path.isdir(picDir):
                os.makedirs(picDir)
            from keras.utils import plot_model
            plot_model(model, to_file=os.path.join(picDir, 'lstm_model.png'), show_shapes='True')

        print ('Fitting model...')
        def lr_scheduler(epoch):
            if epoch % 20 == 0 and epoch != 0:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr*self.params['decay'])
                print("lr changed to {}".format(lr*self.params['decay']))
            return K.get_value(model.optimizer.lr)

        modelPath = os.path.join(modelDir, 'lstm_weights_best.hdf5')
        checkpointer = ModelCheckpoint(filepath=modelPath, monitor='val_acc', verbose=self.verbose, save_best_only=True, mode='max')

        CallBacks = [checkpointer]
        if 'SGD' == self.params['optimizer']:
            scheduler = LearningRateScheduler(lr_scheduler)
            CallBacks.append(scheduler)
        CallBacks.append(EarlyStopping(monitor='val_acc', mode='max', patience=10))

        hist = model.fit(X_train, y_train,
                         batch_size=self.params['batch_size'],
                         epochs=self.params['epochs'],
                         validation_split = 0.2,
                         verbose=self.verbose,
                         callbacks=CallBacks)

        return modelPath

    def prediction(self, X_test, NUM_CLASS, modelPath):
        print ('Predicting results with best model...')
        model = self.create_model(NUM_CLASS)
        model.load_weights(modelPath)
        y_pred = model.predict(X_test)
        return y_pred

    def test(self, X_test, y_test, NUM_CLASS, modelPath):
        print ('Predicting results with best model...')
        model = self.create_model(NUM_CLASS)
        model.load_weights(modelPath)
        score, acc = model.evaluate(X_test, y_test, batch_size=100)

        tmpLine = ['Test score:'+str(score), 'Test accuracy:'+str(acc)]
        content = '\n'.join(tmpLine)
        print(content)
        return acc


def loadData(opts, params):
    X_train_raw, y_train, X_test_raw, y_test, labelMap = loadTrainAndTestData(opts.input, params['data_dim'], opts.dataType)
    NUM_CLASS = len(set(y_test))

    X_train = X_train_raw.reshape(X_train_raw.shape[0], X_train_raw.shape[1], 1)
    X_test = X_test_raw.reshape(X_test_raw.shape[0], X_test_raw.shape[1], 1)
    y_train = np_utils.to_categorical(y_train, NUM_CLASS)
    y_test = np_utils.to_categorical(y_test, NUM_CLASS)

    return X_train, y_train, X_test, y_test, labelMap, NUM_CLASS


def main(opts):
    try:
        PARAMS = generate_default_params()
        X_train, y_train, X_test, y_test, labelMap, NUM_CLASS = loadData(opts, PARAMS)

        lstm = LSTM_Model(opts, PARAMS)
        if opts.testOnly:
            modelPath = opts.testOnly
        else:
            modelPath = lstm.train(X_train, y_train, NUM_CLASS)
        lstm.test(X_test, y_test, NUM_CLASS, modelPath)
    except Exception as e:
        LOG.exception(e)
        raise


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help ='file path of config file')
    parser.add_argument('-d', '--dataType',
                        help ='choose from onlyOrder/both')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help ='verbose or not')
    parser.add_argument('-t', '--testOnly', default='',
                        help ='verbose or not')
    parser.add_argument('-p', '--plotModel', action='store_true',
                        help ='verbose or not')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
