#encoding=utf-8
## copyright@Chenggang Wang
## Email: wang2c9@mail.uc.edu
## All right reserved @@
#
## May 15, 2019

import os
import sys
import argparse
import logging

sys.path.append('/home/carl/work_dir/echo_proj_phase_2/src/bin')
import prepareData
from common_use import Context

if '1' == os.getenv('useGpu'):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import keras.backend as K
from keras.utils import np_utils, plot_model

import numpy as np
import pandas as pd

modelDir = 'modelDir'
if not os.path.isdir(modelDir):
    os.makedirs(modelDir)
LOG = logging.getLogger('modelDir/sae_results')


def generate_default_onlyIncoming_params(dataType):
    if 'both' == dataType:
        return {
            'optimizer': "Adadelta",
            'learning_rate': 0.1,
            'dropout_rate1': 0.1,
            'dropout_rate2': 0,
            'dropout_rate3': 0,
            'dropout_rate4': 0,
            'decay': 0.3,
            'ae_batch_size': 50,
            'ae_epochs': 500,
            'batch_size': 130,
            'epochs': 500,
            'data_dim': 350,
            'encoded1_dim': 330,
            'encoded2_dim': 290,
            'encoded3_dim': 270,
            'encoded4_dim': 250,
            'y_dim': 220,
            'denseLayer': 160,
            'encoded_act1': "elu",
            'encoded_act2': "selu",
            'encoded_act3': "selu",
            'encoded_act4': "softsign",
            'dense_act': "elu",
            'y_act': "tanh",
            'z_act': "tanh"
                }
    else:
        return {
            'optimizer': "Adadelta",
            'learning_rate': 0.005,
            'dropout_rate1': 0.1,
            'dropout_rate2': 0.5,
            'dropout_rate3': 0,
            'dropout_rate4': 0.1,
            'decay': 0.2,
            'ae_batch_size': 70,
            'ae_epochs': 500,
            'batch_size': 90,
            'epochs': 500,
            'data_dim': 600,
            'encoded1_dim': 340,
            'encoded2_dim': 350,
            'encoded3_dim': 330,
            'encoded4_dim': 240,
            'y_dim': 230,
            'denseLayer': 190,
            'encoded_act1': "elu",
            'encoded_act2': "softsign",
            'encoded_act3': "softsign",
            'encoded_act4': "selu",
            'dense_act': "softsign",
            'y_act': "softsign",
            'z_act': "elu"
                }


def generate_default_whole_params(dataType):
    if 'onlyOrder' == dataType:
        return {
                'optimizer': 'Adamax',
                'learning_rate': 0.1,
                'dropout_rate1': 0.3,
                'dropout_rate2': 0,
                'dropout_rate3': 0.1,
                'dropout_rate4': 0.4,
                'decay': 0.2,
                'ae_batch_size': 130,
                'ae_epochs': 500,
                'batch_size': 30,
                'epochs': 500,
                'data_dim': 500,
                'encoded1_dim': 330,
                'encoded2_dim': 260,
                'encoded3_dim': 200,
                'encoded4_dim': 190,
                'y_dim': 160,
                'denseLayer': 170,
                'encoded_act1': 'tanh',
                'encoded_act2': 'elu',
                'encoded_act3': 'tanh',
                'encoded_act4': 'elu',
                'dense_act': 'selu',
                'y_act': 'elu',
                'z_act': 'softsign'
                }
    elif 'both' == dataType:
        return {
                'optimizer': 'Adam',
                'learning_rate': 0.005,
                'dropout_rate1': 0.2,
                'dropout_rate2': 0,
                'dropout_rate3': 0,
                'dropout_rate4': 0.3,
                'decay': 0.3,
                'ae_batch_size': 110,
                'ae_epochs': 500,
                'batch_size': 110,
                'epochs': 500,
                'data_dim': 425,
                'encoded1_dim': 330,
                'encoded2_dim': 260,
                'encoded3_dim': 330,
                'encoded4_dim': 280,
                'y_dim': 250,
                'denseLayer': 130,
                'encoded_act1': 'elu',
                'encoded_act2': 'tanh',
                'encoded_act3': 'selu',
                'encoded_act4': 'elu',
                'dense_act': 'tanh',
                'y_act': 'softsign',
                'z_act': 'elu'
                }
    else:
        raise ValueError('wrong dataType value')


class SAE():
    def __init__(self, opts, params, name='sae'):
        self.inputDim = params['data_dim']
        self.params = params
        self.verbose = opts.verbose
        self.name = name

        outDir = os.path.join(os.getcwd(), 'modelDir/sae_tmp')
        if not os.path.isdir(outDir):
            os.makedirs(outDir)
        self.ae_modelPath = os.path.join(outDir, 'tmp_AeModel.hdf5')

    def encoder(self, inputLayer):
        params = self.params
        print('Setup encoder layers...')
        encoded1 = Dense(params['encoded1_dim'], activation=params['encoded_act1'], kernel_initializer='glorot_normal')(inputLayer)
        batch1 = BatchNormalization()(encoded1)
        dropout1 = Dropout(rate=params['dropout_rate1'])(batch1)

        encoded2 = Dense(params['encoded2_dim'], activation=params['encoded_act2'], kernel_initializer='glorot_normal')(dropout1)
        batch2 = BatchNormalization()(encoded2)
        dropout2 = Dropout(rate=params['dropout_rate2'])(batch2)

        encoded3 = Dense(params['encoded3_dim'], activation=params['encoded_act3'], kernel_initializer='glorot_normal')(dropout2)
        batch3 = BatchNormalization()(encoded3)
        dropout3 = Dropout(rate=params['dropout_rate3'])(batch3)

        encoded4 = Dense(params['encoded4_dim'], activation=params['encoded_act4'], kernel_initializer='glorot_normal')(dropout3)
        batch4 = BatchNormalization()(encoded4)
        dropout4 = Dropout(rate=params['dropout_rate4'])(batch4)

        y = Dense(params['y_dim'], activation=params['y_act'], name='LR', kernel_initializer='glorot_normal')(dropout4)
        batch = BatchNormalization()(y)

        return batch

    def decoder(self, y):
        params = self.params
        print('Setup decoder layers...')
        dropout4 = Dropout(rate=params['dropout_rate4'])(y)
        decoded4 = Dense(params['encoded4_dim'], activation=params['encoded_act4'], kernel_initializer='glorot_normal')(dropout4)
        batch4 = BatchNormalization()(decoded4)

        dropout3 = Dropout(rate=params['dropout_rate3'])(batch4)
        decoded3 = Dense(params['encoded3_dim'], activation=params['encoded_act3'], kernel_initializer='glorot_normal')(dropout3)
        batch3 = BatchNormalization()(decoded3)

        dropout2 = Dropout(rate=params['dropout_rate2'])(batch3)
        decoded2 = Dense(params['encoded2_dim'], activation=params['encoded_act2'], kernel_initializer='glorot_normal')(dropout2)
        batch2 = BatchNormalization()(decoded2)

        dropout1 = Dropout(rate=params['dropout_rate1'])(batch2)
        decoded1 = Dense(params['encoded1_dim'], activation=params['encoded_act1'], kernel_initializer='glorot_normal')(dropout1)
        batch1 = BatchNormalization()(decoded1)

        z = Dense(self.inputDim, activation=params['z_act'], kernel_initializer='glorot_normal')(batch1)

        return z

    def create_autoencoder(self):
        print('create ae model...')
        inputLayer = Input(shape=(self.inputDim, ))
        autoencoder = Model(inputLayer, self.decoder(self.encoder(inputLayer)))
        autoencoder.compile(optimizer=self.params['optimizer'], loss='mse')  # reporting the loss
        return autoencoder

    def build_ae(self, X_train):
        autoencoder = self.create_autoencoder()
        print('Build autoencoder...')

        def lr_scheduler(epoch):
            if epoch % 20 == 0 and epoch != 0:
                lr = K.get_value(autoencoder.optimizer.lr)
                K.set_value(autoencoder.optimizer.lr, lr*self.params['decay'])
                print("lr changed to {}".format(lr*self.params['decay']))
            return K.get_value(autoencoder.optimizer.lr)

        checkpointer = ModelCheckpoint(filepath=self.ae_modelPath, monitor='val_loss', verbose=self.verbose, save_best_only=True, mode='min')

        CallBacks=[checkpointer]
        scheduler = LearningRateScheduler(lr_scheduler)
        CallBacks.append(scheduler)
        CallBacks.append(EarlyStopping(monitor='val_loss', mode='min', patience=6))

        autoencoder_train = autoencoder.fit(X_train, X_train,
                                            epochs=self.params['epochs'],
                                            batch_size=self.params['batch_size'],
                                            validation_split=0.2,
                                            shuffle=True,
                                            callbacks=CallBacks)

        return autoencoder

    def fc(self, enco, outputDim):
        params = self.params
        den = Dense(params['denseLayer'], activation=params['dense_act'], kernel_initializer='glorot_normal')(enco)
        batch_den = BatchNormalization()(den)
        clf = Dense(outputDim, activation='softmax', kernel_initializer='glorot_normal')(batch_den)
        return clf

    def create_model(self, NUM_CLASS):
        inputLayer = Input(shape=(self.inputDim, ))
        full_model = Model(inputLayer, self.fc(self.encoder(inputLayer), NUM_CLASS))
        full_model.compile(loss='categorical_crossentropy',
                           optimizer=self.params['optimizer'],
                           metrics=['accuracy'])
        return full_model

    def plotTheModel(self, NUM_CLASS):
        autoencoder = self.create_autoencoder()
        full_model = self.create_model(NUM_CLASS)
        picDir = os.path.join(modelDir, 'pic')
        if not os.path.isdir(picDir):
            os.makedirs(picDir)
        from keras.utils import plot_model
        plot_model(autoencoder, to_file=os.path.join(picDir, 'autoencoder.png'), show_shapes='True')
        plot_model(full_model, to_file=os.path.join(picDir, 'ae_clf.png'), show_shapes='True')


    def train(self, X_train, y_train, NUM_CLASS):
        autoencoder = self.build_ae(X_train)
        autoencoder.load_weights(self.ae_modelPath)

        full_model = self.create_model(NUM_CLASS)
        LayNum = len(full_model.layers) - 4

        for l1, l2 in zip(full_model.layers[:LayNum], autoencoder.layers[:LayNum]):
            l1.set_weights(l2.get_weights())

        #for layer in full_model.layers[:LayNum]:
        #    layer.trainable = False


        print('start to train classifier...')

        def lr_scheduler(epoch):
            if epoch % 20 == 0 and epoch != 0:
                lr = K.get_value(full_model.optimizer.lr)
                K.set_value(full_model.optimizer.lr, lr*self.params['decay'])
                print("lr changed to {}".format(lr*self.params['decay']))
            return K.get_value(full_model.optimizer.lr)

        clf_modelPath = os.path.join(modelDir, 'sae_weights_best.hdf5')
        checkpointer = ModelCheckpoint(filepath=clf_modelPath, monitor='val_acc', verbose=self.verbose, save_best_only=True, mode='max')

        CallBacks = [checkpointer]
        scheduler = LearningRateScheduler(lr_scheduler)
        CallBacks.append(scheduler)
        CallBacks.append(EarlyStopping(monitor='val_acc', mode='max', patience=6))

        classify_train = full_model.fit(X_train, y_train,
                                        batch_size=self.params['batch_size'],
                                        epochs=self.params['epochs'],
                                        verbose=self.verbose,
                                        shuffle=True,
                                        validation_split=0.2,
                                        callbacks=CallBacks)

        return clf_modelPath

    def test(self, X_test, y_test, NUM_CLASS, clf_modelPath):
        print ('Predicting results with best model...')
        model = self.create_model(NUM_CLASS)
        model.load_weights(clf_modelPath)
        score, acc = model.evaluate(X_test, y_test, batch_size=100)

        print('Test score:', score)
        print('Test accuracy:', acc)

        return acc

    def prediction(self, X_test, NUM_CLASS, modelPath):
        print ('Predicting results with best model...')
        model = self.create_model(NUM_CLASS)
        model.load_weights(modelPath)
        y_pred = model.predict(X_test)
        return y_pred

    def classification_report(self, test_labels, predictions):
        from sklearn.metrics import classification_report
        num_classes = len(set(test_labels))
        target_names = ["Class {}".format(i) for i in range(num_classes)]
        reports = classification_report(test_labels, predictions, target_names=target_names)
        print(reports)


def loadData(opts, PARAMS):
    X_train, y_train, X_test, y_test, labelMap = prepareData.loadTrainAndTestData(opts.input, PARAMS['data_dim'], opts.dataType, opts.mode)
    NUM_CLASS = len(set(y_test))
    y_train = np_utils.to_categorical(y_train, NUM_CLASS)
    y_test = np_utils.to_categorical(y_test, NUM_CLASS)

    return X_train, y_train, X_test, y_test, labelMap, NUM_CLASS


def loadTestData(opts, PARAMS):
    allData, allLabel, labelMap = prepareData.loadData(opts.input, PARAMS['data_dim'], opts.dataType, opts.mode)
    NUM_CLASS = len(set(allLabel))
    allLabel = np_utils.to_categorical(allLabel, NUM_CLASS)

    return allData, allLabel, labelMap, NUM_CLASS


def main(opts):
    try:
        if "onlyIncoming" == opts.mode:
            PARAMS = generate_default_onlyIncoming_params(opts.dataType)
        else:
            PARAMS = generate_default_whole_params(opts.dataType)

        sae = SAE(opts, PARAMS)
        if opts.plotModel:
            sae.plotTheModel(NUM_CLASS)
            return
        if opts.testOnly:
            X_test, y_test, labelMap, NUM_CLASS = loadTestData(opts, PARAMS)
            modelPath = opts.testOnly
        else:
            X_train, y_train, X_test, y_test, labelMap, NUM_CLASS = loadData(opts, PARAMS)
            modelPath = sae.train(X_train, y_train, NUM_CLASS)
        acc = sae.test(X_test, y_test, NUM_CLASS, modelPath)
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
                        help ='verbose or not')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help ='verbose or not')
    parser.add_argument('-p', '--plotModel', action='store_true',
                        help ='verbose or not')
    opts = parser.parse_args()
    return opts



if __name__=="__main__":
    opts = parseOpts(sys.argv)
    main(opts)
