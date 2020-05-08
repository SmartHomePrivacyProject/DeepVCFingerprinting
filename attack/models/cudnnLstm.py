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

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from keras.layers import Dense, Dropout, CuDNNLSTM, Embedding, Flatten, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import keras.backend as K
from keras.utils import np_utils
import pandas as pd
import numpy as np

sys.path.append('bin')
import prepareData

modelDir = 'modelDir'
if not os.path.isdir(modelDir):
    os.makedirs(modelDir)
LOG = logging.getLogger('modelDir/cudnnLstm_results')


def generate_default_onlyIncoming_params(dataType):
    if 'both' == dataType:
        return {
                'optimizer': "Adamax",
                'learning_rate': 0.1,
                'dropout_rate1': 0,
                'dropout_rate2': 0.1,
                'dropout_rate3': 0,
                'dropout_rate4': 0.1,
                'dropout_rate5': 0.5,
                'batch_size': 170,
                'epochs': 500,
                'decay': 0.1944,
                'data_dim': 500,
                'layer1': 170,
                'layer2': 290,
                'layer3': 170,
                'layer4': 90,
                'layer5': 250,
                'dense1': 80,
                'dense2': 150,
                'dense1_act': "elu",
                'dense2_act': "elu",
                'kernel_ini': 'glorot_normal'
                }
    else:
        return {
                'optimizer': "SGD",
                'learning_rate': 0.05,
                'dropout_rate1': 0.1,
                'dropout_rate2': 0.4,
                'dropout_rate3': 0.3,
                'dropout_rate4': 0.1,
                'dropout_rate5': 0.5,
                'batch_size': 130,
                'epochs': 500,
                'decay': 0.2,
                'data_dim': 575,
                'layer1': 90,
                'layer2': 90,
                'layer3': 210,
                'layer4': 110,
                'layer5': 230,
                'dense1': 100,
                'dense2': 160,
                'dense1_act': "elu",
                'dense2_act': "softsign",
                'kernel_ini': 'glorot_normal'
                }


def generate_default_whole_params(dataType):
    if 'onlyOrder' == dataType:
        return {
                'optimizer': 'Adam',
                'learning_rate': 0.005,
                'dropout_rate1': 0.1,
                'dropout_rate2': 0.4,
                'dropout_rate3': 0.1,
                'dropout_rate4': 0.1,
                'dropout_rate5': 0.4,
                'batch_size': 70,
                'decay': 0.1944,
                'epochs': 500,
                'data_dim': 425,
                'layer1': 230,
                'layer2': 310,
                'layer3': 150,
                'layer4': 250,
                'layer5': 110,
                'dense1': 90,
                'dense2': 70,
                'dense1_act': 'selu',
                'dense2_act': 'selu',
                'kernel_ini': 'glorot_normal'
                }
    elif 'both' == dataType:
        return {
                'optimizer': 'Adamax',
                'dropout_rate1': 0.4,
                'dropout_rate2': 0.1,
                'dropout_rate3': 0.1,
                'dropout_rate4': 0.3,
                'dropout_rate5': 0.5,
                'batch_size': 130,
                'decay': 0.1944,
                'epochs': 500,
                'data_dim': 350,
                'layer1': 210,
                'layer2': 190,
                'layer3': 190,
                'layer4': 190,
                'layer5': 110,
                'dense1': 70,
                'dense2': 70,
                'dense1_act': 'selu',
                'dense2_act': 'selu',
                'kernel_ini': 'glorot_normal'
                }
    else:
        raise ValueError('datatype value is wrong')


class LSTM():
    def __init__(self, opts, params, name='cudnnLstm'):
        self.params = params
        self.verbose = opts.verbose
        self.plotModel = opts.plotModel
        self.name = name


    def create_model(self, NUM_CLASS):
        print ('Creating model...')
        model = Sequential()
        model.add(CuDNNLSTM(units=self.params['layer1'], input_shape=(self.params['data_dim'], 1), return_sequences=True, kernel_initializer=self.params['kernel_ini']))
        model.add(Dropout(rate=self.params['dropout_rate1']))

        model.add(CuDNNLSTM(units=self.params['layer2'], return_sequences=True, kernel_initializer=self.params['kernel_ini']))
        model.add(Dropout(rate=self.params['dropout_rate2']))

        model.add(CuDNNLSTM(units=self.params['layer3'], return_sequences=True, kernel_initializer=self.params['kernel_ini']))
        model.add(Dropout(rate=self.params['dropout_rate3']))

        model.add(CuDNNLSTM(units=self.params['layer4'], return_sequences=True, kernel_initializer=self.params['kernel_ini']))
        model.add(Dropout(rate=self.params['dropout_rate4']))

        model.add(CuDNNLSTM(units=self.params['layer5'], return_sequences=True, kernel_initializer=self.params['kernel_ini']))
        model.add(Dropout(rate=self.params['dropout_rate5']))

        model.add(Flatten())
        model.add(Dense(self.params['dense1'], activation=self.params['dense1_act'], kernel_initializer=self.params['kernel_ini']))
        model.add(BatchNormalization())
        model.add(Dense(self.params['dense2'], activation=self.params['dense2_act'], kernel_initializer=self.params['kernel_ini']))
        model.add(BatchNormalization())
        model.add(Dense(NUM_CLASS, activation='softmax', kernel_initializer=self.params['kernel_ini']))

        print ('Compiling...')
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.params['optimizer'],
                      metrics=['accuracy'])
        return model


    def train(self, X_train, y_train, NUM_CLASS):
        model = self.create_model(NUM_CLASS)
        print ('Fitting model...')
        if self.plotModel:
            from keras.utils import plot_model
            plot_model(model, to_file='lstm.png', show_shapes=False)

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
        CallBacks.append(EarlyStopping(monitor='val_acc', mode='max', patience=6))

        hist = model.fit(X_train, y_train,
                         shuffle=True,
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
    X_train_raw, y_train, X_test_raw, y_test, labelMap = prepareData.loadTrainAndTestData(opts.input, params['data_dim'], opts.dataType, opts.mode)
    NUM_CLASS = len(set(y_test))
    X_train = X_train_raw.reshape(X_train_raw.shape[0], X_train_raw.shape[1], 1)
    X_test = X_test_raw.reshape(X_test_raw.shape[0], X_test_raw.shape[1], 1)
    y_train = np_utils.to_categorical(y_train, NUM_CLASS)
    y_test = np_utils.to_categorical(y_test, NUM_CLASS)

    return X_train, y_train, X_test, y_test, labelMap, NUM_CLASS


def loadTestData(opts, params):
    allData_raw, allLabel_raw, labelMap = prepareData.loadData(opts.input, params['data_dim'], opts.dataType, opts.mode)
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
        lstm = LSTM(opts, PARAMS)

        if opts.testOnly:
            X_test, y_test, labelMap, NUM_CLASS = loadTestData(opts, PARAMS)
            modelPath = opts.testOnly
        else:
            X_train, y_train, X_test, y_test, labelMap, NUM_CLASS = loadData(opts, PARAMS)
            modelPath = lstm.train(X_train, y_train, NUM_CLASS)
        acc = lstm.test(X_test, y_test, NUM_CLASS, modelPath)
        if opts.output:
            tmp = 'LSTM with data {} accuracy and dataType {} has an accuracy: {:f}'.format(opts.input, opts.dataType, acc)
            import nFold
            outfile = os.path.join(nFold.ResDir, opts.output)
            print('file save to {}'.format(outfile))
            with open(outfile, 'w') as f:
                f.write(tmp+'\n')
    except Exception as e:
        LOG.exception(e)
        raise


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help ='file path of config file')
    parser.add_argument('-d', '--dataType',
                        help ='choose from onlyOrder/both')
    parser.add_argument('-o', '--output', default='',
                        help ='output file name')
    parser.add_argument('-m', '--mode', default='',
                        help ='output file name')
    parser.add_argument('-t', '--testOnly', default='',
                        help ='choose from onlyOrder/both')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help ='verbose or not')
    parser.add_argument('-p', '--plotModel', action='store_true',
                        help ='verbose or not')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
