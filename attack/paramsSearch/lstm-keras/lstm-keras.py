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


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
#sess = tf.Session(config=config)

from keras.layers import Dense, Dropout, CuDNNLSTM, Embedding, Flatten, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, EarlyStopping
import keras.backend as K
import pandas as pd
import numpy as np
from keras.utils import np_utils

import nni

sys.path.append('/home/carl007/work_dir/echo_proj_phase_2/src/bin')
from prepareData import loadTrainAndTestData
import cudnnLstm
import basicSetting


LOG, modelDir = basicSetting.getDirs()


class SendMetrics(Callback):
    def on_epoch_end(self, epoch, logs={}):
        LOG.debug(logs)
        nni.report_intermediate_result(logs["val_acc"])


class LSTM():
    def __init__(self, opts, params, name='lstm'):
        self.verbose = opts.verbose
        self.lstm = cudnnLstm.LSTM(opts, params)

    def train(self, params, X_train, y_train, NUM_CLASS):
        model = self.lstm.create_model(NUM_CLASS)
        if opts.plotModel:
            from keras.utils import plot_model
            plot_model(model, to_file='modelDir/lstm_model.png', show_shapes='True')

        print ('Fitting model...')

        def lr_scheduler(epoch):
            if epoch % 20 == 0 and epoch != 0:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr*params['decay'])
                print("lr changed to {}".format(lr*params['decay']))
            return K.get_value(model.optimizer.lr)

        modelPath = os.path.join(modelDir, 'lstm_weights_best.hdf5')
        checkpointer = ModelCheckpoint(filepath=modelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        myCallBacks = [checkpointer]
        if params['optimizer'] == 'SGD':
            scheduler = LearningRateScheduler(lr_scheduler)
            myCallBacks.append(scheduler)
        myCallBacks.append(SendMetrics())
        myCallBacks.append(EarlyStopping(monitor='val_acc', mode='max', patience=10))

        hist = model.fit(X_train, y_train,
                         batch_size=params['batch_size'],
                         epochs=params['epochs'],
                         validation_split = 0.2,
                         verbose=opts.verbose,
                         callbacks=myCallBacks)

        return modelPath

    def test(self, X_test, y_test, NUM_CLASS, modelPath):
        print ('Predicting results with best model...')
        model = self.lstm.create_model(NUM_CLASS)
        model.load_weights(modelPath)
        score, acc = model.evaluate(X_test, y_test, batch_size=100)

        nni.report_final_result(acc)

        print('Test score:', score)
        print('Test accuracy:', acc)


def main(opts):
    RECEIVED_PARAMS = nni.get_next_parameter()
    LOG.debug(RECEIVED_PARAMS)
    if 'onlyIncoming' == opts.mode:
        PARAMS = cudnnLstm.generate_default_onlyIncoming_params(opts.dataType)
    elif 'both' == opts.mode:
        PARAMS = cudnnLstm.generate_default_whole_params(opts.dataType)
    PARAMS.update(RECEIVED_PARAMS)
    try:
        X_train_raw, y_train, X_test_raw, y_test, labelMap = loadTrainAndTestData(opts.input, PARAMS['data_dim'], opts.dataType, opts.mode)
        NUM_CLASS = len(set(y_test))

        X_train = X_train_raw.reshape(X_train_raw.shape[0], X_train_raw.shape[1], 1)
        X_test = X_test_raw.reshape(X_test_raw.shape[0], X_test_raw.shape[1], 1)
        y_train = np_utils.to_categorical(y_train, NUM_CLASS)
        y_test = np_utils.to_categorical(y_test, NUM_CLASS)

        lstm_model = LSTM(opts, PARAMS)
        modelPath = lstm_model.train(PARAMS, X_train, y_train, NUM_CLASS)
        lstm_model.test(X_test, y_test, NUM_CLASS, modelPath)
    except Exception as e:
        LOG.exception(e)
        raise


if __name__ == "__main__":
    opts = cudnnLstm.parseOpts(sys.argv)
    main(opts)
