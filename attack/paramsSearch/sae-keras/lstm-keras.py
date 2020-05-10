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

import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout, LSTM, Embedding, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
import keras.backend as K
import pandas as pd
import numpy as np
from keras.utils import np_utils

import nni

sys.path.append('bin')
from prepareData import loadTrainAndTestData
from common_use import Context

if not os.path.isdir('results'):
    os.makedirs('results')
LOG = logging.getLogger('results/lstm_results')


def setUpContext(opts):
    dpath = opts.input
    data_dim = int(opts.dim)
    epochs = opts.epochs
    batch_size = opts.batch_size
    verbose = opts.verbose
    context = Context(dpath, data_dim, epochs, 0, batch_size, verbose)
    return context


#model.add(LSTM(units=175, activation='relu', input_shape=(1, 256),
#               recurrent_activation='hard_sigmoid', return_sequences=True))
def create_model(context, params):
    print ('Creating model...')
    model = Sequential()
    model.add(LSTM(units=context.data_dim, activation=params['activation'], input_shape=(1,context.data_dim),
                   recurrent_activation='hard_sigmoid', return_sequences=True))
    model.add(LSTM(units=300, activation=params['activation'],
                   recurrent_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(params['dropout_rate']))
    model.add(LSTM(units=250, activation=params['activation'],
                   recurrent_activation='hard_sigmoid', return_sequences=True))
    model.add(LSTM(units=200, activation=params['activation'],
                   recurrent_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(params['dropout_rate']))
    model.add(LSTM(units=175, activation=params['activation'],
                   recurrent_activation='hard_sigmoid', return_sequences=True))
    model.add(LSTM(units=150, activation=params['activation'],
                   recurrent_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(params['dropout_rate']))
    model.add(LSTM(units=125, activation=params['activation'],
                   recurrent_activation='hard_sigmoid'))
    #model.add(Flatten())
    model.add(Dense(100, activation='softmax'))

    print ('Compiling...')
    model.compile(loss='categorical_crossentropy',
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    return model


class SendMetrics(Callback):
    def on_epoch_end(self, epoch, logs={}):
        LOG.debug(logs)
        nni.report_intermediate_result(logs["val_acc"])


def generate_default_params():
    return {
            'optimizer': 'Adam',
            'learning_rate': 0.001,
            'activation': 'relu',
            'dropout_rate': 0.5
            }


def train(context, params):
    X_train_raw, y_train, X_test_raw, y_test, labelMap = loadTrainAndTestData(context.dpath,
                                                                          context.data_dim)

    X_train = X_train_raw.reshape(X_train_raw.shape[0], 1, X_train_raw.shape[1])
    X_test = X_test_raw.reshape(X_test_raw.shape[0], 1, X_test_raw.shape[1])
    y_train = np_utils.to_categorical(y_train, 100)
    y_test = np_utils.to_categorical(y_test, 100)
    #import pdb
    #pdb.set_trace()
    NUM_CLASS = len(y_train)
    model = create_model(context, params)

    print ('Fitting model...')
    modelDir = 'modelDir'

    if not os.path.isdir(modelDir):
        os.makedirs(modelDir)
    modelPath = os.path.join(modelDir, 'lstm_weights_best.hdf5')

    def lr_scheduler(epoch):
        if epoch % 20 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr*0.1)
            print("lr changed to {}".format(lr*0.1))
        return K.get_value(model.optimizer.lr)

    checkpointer = ModelCheckpoint(filepath=modelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    scheduler = LearningRateScheduler(lr_scheduler)

    hist = model.fit(X_train, y_train, batch_size=context.batch_size,
                     epochs=context.epochs1, validation_split = 0.1,
                     verbose=context.verbose, callbacks=[checkpointer, scheduler])

    #from keras.utils import plot_model
    #plot_model(model, to_file='model.png', show_shapes='True')

    return modelPath, X_test, y_test


def test(context, params, modelPath, X_test, y_test):
    print ('Predicting results with best model...')
    model = create_model(context, params)
    model.load_weights(modelPath)
    score, acc = model.evaluate(X_test, y_test, batch_size=100)

    nni.report_final_result(acc)

    print('Test score:', score)
    print('Test accuracy:', acc)


def main(opts):
    context = setUpContext(opts)
    try:
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = generate_default_params()
        PARAMS.update(RECEIVED_PARAMS)
        modelPath, X_test, y_test = train(context, PARAMS)
        test(context, PARAMS, modelPath, X_test, y_test)
    except Exception as e:
        LOG.exception(e)
        raise


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help ='file path of config file')
    parser.add_argument('-d', '--dim',
                        help ='dimension of data sample')
    parser.add_argument('-o', '--output',
                        help ='file path to store result file')
    parser.add_argument('-e', '--epochs', default=100,
                        help ='epochs number')
    parser.add_argument('-b', '--batch_size', default=100,
                        help ='batch_size number')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help ='verbose or not')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
