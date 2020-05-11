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

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, EarlyStopping
import keras.backend as K
from keras.utils import np_utils, plot_model
from tensorflow.python.ops.variables import trainable_variables

import nni

modelsDir = os.getenv('MODELS_DIR')
toolsDir = os.getenv('TOOLS_DIR')
sys.path.append(modelsDir)
sys.path.append(toolsDir)
import prepareData
from common_use import Context
import sae
import basicSetting


LOG, modelDir = basicSetting.getDirs()


class SendMetrics(Callback):
    def on_epoch_end(self, epoch, logs={}):
        LOG.debug(logs)
        nni.report_intermediate_result(logs["val_acc"])


class SAE():
    def __init__(self, opts, params, networkName='SAE'):
        self.networkName = networkName
        self.verbose = opts.verbose
        self.sae_model = sae.SAE(opts, params)

    def train(self, params, X_train, y_train, outputDim):
        LayNum = 5
        autoencoder = self.sae_model.build_ae(X_train)
        autoencoder.load_weights(self.sae_model.ae_modelPath)

        full_model = self.sae_model.create_model(outputDim)

        for l1, l2 in zip(full_model.layers[:LayNum], autoencoder.layers[:LayNum]):
            l1.set_weights(l2.get_weights())

        #for layer in full_model.layers[:LayNum]:
        #    layer.trainable = False


        print('start to train classifier...')

        def lr_scheduler(epoch):
            if epoch % 20 == 0 and epoch != 0:
                lr = K.get_value(full_model.optimizer.lr)
                K.set_value(full_model.optimizer.lr, lr*params['decay'])
                print("lr changed to {}".format(lr*params['decay']))
            return K.get_value(full_model.optimizer.lr)

        clf_modelPath = os.path.join(modelDir, 'sae_weights_best.hdf5')
        checkpointer = ModelCheckpoint(filepath=clf_modelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        CallBacks = [SendMetrics(), checkpointer]
        scheduler = LearningRateScheduler(lr_scheduler)
        CallBacks.append(scheduler)
        CallBacks.append(EarlyStopping(monitor='val_acc', mode='max', patience=6))

        classify_train = full_model.fit(X_train, y_train,
                                        batch_size=params['batch_size'],
                                        epochs=params['epochs'],
                                        shuffle=True,
                                        verbose=self.verbose,
                                        validation_split=0.2,
                                        callbacks=CallBacks)

        return clf_modelPath

    def test(self, X_test, y_test, outputDim, clf_modelPath):
        print ('Predicting results with best model...')
        model = self.sae_model.create_model(outputDim)
        model.load_weights(clf_modelPath)
        score, acc = model.evaluate(X_test, y_test, batch_size=100)

        nni.report_final_result(acc)

        rtnMsg = ['Test score:' + str(score), 'Test accuracy:' + str(acc)]
        print(rtnMsg)

    def classification_report(self, test_labels, predictions):
        from sklearn.metrics import classification_report
        num_classes = len(set(test_labels))
        target_names = ["Class {}".format(i) for i in range(num_classes)]
        reports = classification_report(test_labels, predictions, target_names=target_names)
        print(reports)

    def plotModel(self, model, output):
        from keras.utils import plot_model
        plot_model(self.autoencoder, to_file=os.path.join(output, 'autoencoder.png'), show_shapes='True')
        plot_model(modelFile, to_file='clf.png', show_shapes='True')


def main(opts):
    RECEIVED_PARAMS = nni.get_next_parameter()
    LOG.debug(RECEIVED_PARAMS)
    if 'onlyIncoming' == opts.mode:
        PARAMS = sae.generate_default_onlyIncoming_params(opts.dataType)
    elif 'both' == opts.mode:
        PARAMS = sae.generate_default_whole_params(opts.dataType)
    PARAMS.update(RECEIVED_PARAMS)
    try:
        X_train, y_train, X_test, y_test, labelMap = prepareData.loadTrainAndTestData(opts.input, PARAMS['data_dim'], opts.dataType, opts.mode)
        NUM_CLASS = len(set(y_test))

        y_train = np_utils.to_categorical(y_train, NUM_CLASS)
        y_test = np_utils.to_categorical(y_test, NUM_CLASS)

        sae_model = SAE(opts, PARAMS)
        modelPath = sae_model.train(PARAMS, X_train, y_train, NUM_CLASS)
        sae_model.test(X_test, y_test, NUM_CLASS, modelPath)
    except Exception as e:
        LOG.exception(e)
        raise


if __name__=="__main__":
    opts = sae.parseOpts(sys.argv)
    main(opts)
