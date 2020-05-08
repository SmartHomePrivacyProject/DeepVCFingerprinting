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

if '1' == os.getenv('useGpu'):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
from keras.layers.convolutional import Conv1D
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, BatchNormalization, GlobalAveragePooling1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
import keras.backend as K
from keras.utils import np_utils

import numpy as np
import pandas as pd

sys.path.append('bin')
import prepareData

modelDir = 'modelDir'
if not os.path.isdir(modelDir):
    os.makedirs(modelDir)
LOG = logging.getLogger('modelDir/cnn_results')


def generate_default_onlyIncoming_params(dataType):
    if 'both' == dataType:
        return {
            'optimizer':"Adam",
            'learning_rate':0.05,
            'activation1':"tanh",
            'activation2':"selu",
            'activation3':"elu",
            'activation4':"selu",
            'drop_rate1':0.2,
            'drop_rate2':0.1,
            'drop_rate3':0.4,
            'drop_rate4':0.5,
            'decay':0.5,
            'batch_size':150,
            'data_dim':450,
            'epochs': 500,
            'conv1':256,
            'conv2':32,
            'conv3':128,
            'conv4':32,
            'pool1':3,
            'pool2':2,
            'pool3':1,
            'pool4':2,
            'kernel_size1':9,
            'kernel_size2':9,
            'kernel_size3':11,
            'kernel_size4':15,
            'dense1':120,
            'dense2':140,
            'dense1_act':"selu",
            'dense2_act':"selu"
                }
    else:
        return {
            'optimizer':"SGD",
            'learning_rate':0.05,
            'activation1':"tanh",
            'activation2':"selu",
            'activation3':"softsign",
            'activation4':"selu",
            'drop_rate1':0.2,
            'drop_rate2':0.1,
            'drop_rate3':0.1,
            'drop_rate4':0.2,
            'decay':0.2,
            'batch_size':70,
            'data_dim':550,
            'epochs': 500,
            'conv1':128,
            'conv2':16,
            'conv3':8,
            'conv4':256,
            'pool1':4,
            'pool2':3,
            'pool3':2,
            'pool4':1,
            'kernel_size1':7,
            'kernel_size2':5,
            'kernel_size3':7,
            'kernel_size4':7,
            'dense1':120,
            'dense2':170,
            'dense1_act':"softsign",
            'dense2_act':"elu"
                }


def generate_default_whole_params(dataType):
    if 'onlyOrder' == dataType:
        return {
                'optimizer': 'Adam',
                'learning_rate': 0.01,
                'activation1': 'softsign',
                'activation2': 'softsign',
                'activation3': 'selu',
                'activation4': 'selu',
                'drop_rate1': 0.3,
                'drop_rate2': 0.1,
                'drop_rate3': 0.3,
                'drop_rate4': 0.5,
                'decay': 0.1,
                'batch_size': 70,
                'data_dim': 600,
                'epochs': 500,
                'conv1': 64,
                'conv2': 128,
                'conv3': 256,
                'conv4': 128,
                'pool1': 5,
                'pool2': 3,
                'pool3': 1,
                'pool4': 3,
                'kernel_size1': 15,
                'kernel_size2': 21,
                'kernel_size3': 15,
                'kernel_size4': 11,
                'dense1': 150,
                'dense2': 130,
                'dense1_act': 'selu',
                'dense2_act': 'softsign'
                }
    elif 'both' == dataType:
        return {
                'optimizer': 'Adamax',
                'learning_rate': 0.05,
                'activation1': 'tanh',
                'activation2': 'elu',
                'activation3': 'elu',
                'activation4': 'selu',
                'drop_rate1': 0.1,
                'drop_rate2': 0.3,
                'drop_rate3': 0.1,
                'drop_rate4': 0,
                'decay': 0.13,
                'batch_size': 70,
                'data_dim': 475,
                'epochs': 500,
                'conv1': 128,
                'conv2': 128,
                'conv3': 64,
                'conv4': 256,
                'pool1': 1,
                'pool2': 1,
                'pool3': 1,
                'pool4': 1,
                'kernel_size1': 7,
                'kernel_size2': 19,
                'kernel_size3': 13,
                'kernel_size4': 23,
                'dense1': 180,
                'dense2': 150,
                'dense1_act': 'selu',
                'dense2_act': 'selu'
                }
    else:
        raise ValueError('wrong datatype')


class CNN():
    def __init__(self, opts, params, name='cnn'):
        self.params = params
        self.verbose = opts.verbose
        self.plotModel = opts.plotModel
        self.name = name


    def create_model(self, NUM_CLASS):
        print ('Creating model...')

        layers = [Conv1D(self.params['conv1'], kernel_size=self.params['kernel_size1'], activation=self.params['activation1'], input_shape=(self.params['data_dim'], 1), use_bias=False, kernel_initializer='glorot_normal'),
                  BatchNormalization(),
                  MaxPooling1D(self.params['pool1']),
                  Dropout(rate=self.params['drop_rate1']),

                  Conv1D(self.params['conv2'], kernel_size=self.params['kernel_size2'], activation=self.params['activation2'], kernel_initializer='glorot_normal'),
                  BatchNormalization(),
                  MaxPooling1D(self.params['pool2']),
                  Dropout(rate=self.params['drop_rate2']),

                  Conv1D(self.params['conv3'], kernel_size=self.params['kernel_size3'], activation=self.params['activation3'], kernel_initializer='glorot_normal'),
                  BatchNormalization(),
                  MaxPooling1D(self.params['pool3']),
                  Dropout(rate=self.params['drop_rate3']),

                  Conv1D(self.params['conv4'], kernel_size=self.params['kernel_size4'], activation=self.params['activation4'], kernel_initializer='glorot_normal'),
                  BatchNormalization(),
                  MaxPooling1D(self.params['pool4']),
                  GlobalAveragePooling1D(),


                  Dense(self.params['dense1'], activation=self.params['dense1_act'], kernel_initializer='glorot_normal'),
                  BatchNormalization(),
                  Dense(self.params['dense2'], activation=self.params['dense2_act'], kernel_initializer='glorot_normal'),
                  BatchNormalization(),
                  Dense(NUM_CLASS, activation='softmax')]

        model = Sequential(layers)

        print ('Compiling...')
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.params['optimizer'],
                      metrics=['accuracy'])
        return model


    def train(self, X_train, y_train, NUM_CLASS):
        '''train the cnn model'''
        model = self.create_model(NUM_CLASS)
        if self.plotModel:
            picDir = os.path.join(modelDir, 'pic')
            if not os.path.isdir(picDir):
                os.makedirs(picDir)
            picPath = os.path.join(picDir, 'cnn_model.png')
            from keras.utils import plot_model
            plot_model(model, to_file=picPath, show_shapes='True')

        print ('Fitting model...')

        def lr_scheduler(epoch):
            if epoch % 20 == 0 and epoch != 0:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr*self.params['decay'])
                print("lr changed to {}".format(lr*self.params['decay']))
            return K.get_value(model.optimizer.lr)

        modelPath = os.path.join(modelDir, 'cnn_weights_best.hdf5')
        checkpointer = ModelCheckpoint(filepath=modelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        CallBacks = [checkpointer]
        if self.params['optimizer'] == 'SGD':
            scheduler = LearningRateScheduler(lr_scheduler)
            CallBacks.append(scheduler)
        CallBacks.append(EarlyStopping(monitor='val_acc', mode='max', patience=6))
        #CallBacks.append(TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True))

        hist = model.fit(X_train, y_train,
                         batch_size=self.params['batch_size'],
                         epochs=self.params['epochs'],
                         validation_split = 0.2,
                         verbose=self.verbose,
                         callbacks=CallBacks)

        if self.plotModel:
            from keras.utils import plot_model
            plot_model(model, to_file='model.png', show_shapes='True')

        return modelPath

    def prediction(self, X_test, NUM_CLASS, modelPath):
        print ('Predicting results with best model...')
        model = self.create_model(NUM_CLASS)
        model.load_weights(modelPath)
        y_pred = model.predict(X_test)
        return y_pred

    def test(self, X_test, y_test, NUM_CLASS, modelPath):
        print ('Testing with best model...')
        model = self.create_model(NUM_CLASS)
        model.load_weights(modelPath)
        score, acc = model.evaluate(X_test, y_test, batch_size=100)

        print('Test score:', score)
        print('Test accuracy:', acc)
        return acc


def loadData(opts, PARAMS):
    X_train_raw, y_train, X_test_raw, y_test, labelMap = prepareData.loadTrainAndTestData(opts.input, PARAMS['data_dim'], opts.dataType, opts.mode)
    NUM_CLASS = len(set(y_test))
    X_train = X_train_raw.reshape(X_train_raw.shape[0], X_train_raw.shape[1], 1)
    X_test = X_test_raw.reshape(X_test_raw.shape[0], X_test_raw.shape[1], 1)
    y_train = np_utils.to_categorical(y_train, NUM_CLASS)
    y_test = np_utils.to_categorical(y_test, NUM_CLASS)

    return X_train, y_train, X_test, y_test, labelMap, NUM_CLASS

def loadTestData(opts, params):
    allData_raw, allLabel_raw, labelMap = prepareData.loadData(opts.input, params['data_dim'], opts.dataType, mode=opts.mode)
    NUM_CLASS = len(set(allLabel_raw))
    allData = allData_raw.reshape(allData_raw.shape[0], allData_raw.shape[1], 1)
    allLabel = np_utils.to_categorical(allLabel_raw, NUM_CLASS)

    return allData, allLabel, labelMap, NUM_CLASS

def main(opts):
    try:
        if 'onlyIncoming' == opts.mode:
            PARAMS = generate_default_onlyIncoming_params(opts.dataType)
        else:
            PARAMS = generate_default_whole_params(opts.dataType)
        model = CNN(opts, PARAMS)

        if not opts.testOnly:
            X_train, y_train, X_test, y_test, labelMap, NUM_CLASS = loadData(opts, PARAMS)
            modelPath = model.train(X_train, y_train, NUM_CLASS)
        else:
            modelPath = opts.testOnly
            X_test, y_test, labelMap, NUM_CLASS = loadTestData(opts, PARAMS)
        acc = model.test(X_test, y_test, NUM_CLASS, modelPath)
        if opts.output:
            tmp = 'cnn with data {} accuracy and dataType {} has an accuracy: {:f}'.format(opts.input, opts.dataType, acc)
            import nFold
            outfile = os.path.join(nFold.ResDir, opts.output)
            print('file save to {}'.format(outfile))
            with open(outfile, 'w') as f:
                f.write(tmp+'\n')
    except Exception as e:
        LOG.exception(e)
        raise

class Opts():
    def __init__(self, useGpu=False, verbose=True, plotModel=False):
        self.useGpu = useGpu
        self.verbose = verbose
        self.plotModel = plotModel

def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help ='file path of config file')
    parser.add_argument('-d', '--dataType',
                        help ='dataType to use, onlyOrder/both')
    parser.add_argument('-o', '--output', default='',
                        help ='output file name')
    parser.add_argument('-m', '--mode', default='',
                        help ='output file name')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help ='verbose or not')
    parser.add_argument('-t', '--testOnly', default='',
                        help ='only run test with given model')
    parser.add_argument('-p', '--plotModel', action='store_true',
                        help ='verbose or not')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
