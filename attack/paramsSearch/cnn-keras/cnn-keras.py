#encoding=utf-8
# Copyright@Chenggang Wang
# 1277223029@qq.com
# All right reserved
#
# May 27, 2019

import os
import sys
import argparse
import logging

import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

from keras.initializers import glorot_normal
from keras.layers.convolutional import Conv1D
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, BatchNormalization, GlobalAveragePooling1D, Activation
from keras.models import Sequential
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, EarlyStopping
import keras.backend as K
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split

import nni

sys.path.append('/home/carl007/work_dir/echo_proj_phase_2/src/bin')
import prepareData
import cnn
import basicSetting


LOG, modelDir = basicSetting.getDirs()


class SendMetrics(Callback):
    def on_epoch_end(self, epoch, logs={}):
        LOG.debug(logs)
        nni.report_intermediate_result(logs["val_acc"])


class CNN():
    def __init__(self, opts, params, modelName='CNN'):
        self.verbose = opts.verbose
        self.cnn = cnn.CNN(opts, params)

    def train(self, params, X_train, y_train, NUM_CLASS):
        '''train the cnn model'''
        model = self.cnn.create_model(NUM_CLASS)

        print ('Fitting model...')

        def lr_scheduler(epoch):
            if epoch % 20 == 0 and epoch != 0:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr*params['decay'])
                print("lr changed to {}".format(lr*params['decay']))
            return K.get_value(model.optimizer.lr)

        modelPath = os.path.join(modelDir, 'cnn_weights_best.hdf5')
        checkpointer = ModelCheckpoint(filepath=modelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        scheduler = LearningRateScheduler(lr_scheduler)
        earlyStopper = EarlyStopping(monitor='val_acc', mode='max', patience=6)
        myCallbacks=[SendMetrics(), checkpointer, scheduler, earlyStopper]

        hist = model.fit(X_train, y_train,
                         batch_size=params['batch_size'],
                         epochs=params['epochs'],
                         validation_split = 0.2,
                         verbose=opts.verbose,
                         callbacks=myCallbacks)

        #from keras.utils import plot_model
        #plot_model(model, to_file='model.png', show_shapes='True')

        return modelPath

    def test(self, X_test, y_test, NUM_CLASS, modelPath):
        print ('Predicting results with best model...')
        model = self.cnn.create_model(NUM_CLASS)
        model.load_weights(modelPath)
        score, acc = model.evaluate(X_test, y_test, batch_size=100)

        nni.report_final_result(acc)

        print('Test score:', score)
        print('Test accuracy:', acc)


def main(opts):
    RECEIVED_PARAMS = nni.get_next_parameter()
    LOG.debug(RECEIVED_PARAMS)
    if 'onlyIncoming' == opts.mode:
        PARAMS = cnn.generate_default_onlyIncoming_params(opts.dataType)
    elif 'both' == opts.mode:
        PARAMS = cnn.generate_default_whole_params(opts.dataType)
    PARAMS.update(RECEIVED_PARAMS)
    try:
        X_train_raw, y_train, X_test_raw, y_test, labelMap = prepareData.loadTrainAndTestData(opts.input, PARAMS['data_dim'], opts.dataType, mode=opts.mode)
        NUM_CLASS = len(set(y_test))
        X_train = X_train_raw.reshape(X_train_raw.shape[0], X_train_raw.shape[1], 1)
        X_test = X_test_raw.reshape(X_test_raw.shape[0], X_test_raw.shape[1], 1)
        y_train = np_utils.to_categorical(y_train, NUM_CLASS)
        y_test = np_utils.to_categorical(y_test, NUM_CLASS)

        cnn_model = CNN(opts, PARAMS)
        modelPath  = cnn_model.train(PARAMS, X_train, y_train, NUM_CLASS)
        cnn_model.test(X_test, y_test, NUM_CLASS, modelPath)
    except Exception as e:
        LOG.exception(e)
        raise


if __name__ == "__main__":
    opts = cnn.parseOpts(sys.argv)
    main(opts)
