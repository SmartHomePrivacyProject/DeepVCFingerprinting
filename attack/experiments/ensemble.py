#!/usr/bin/env python3.6

import os
import sys
import argparse
import numpy as np
import pandas as pd
import time
import re

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from statistics import mean
from collections import defaultdict

if '1' == os.getenv('useGpu'):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
from keras.utils import np_utils


modelsDir = os.getenv('MODELS_DIR')
toolsDir = os.getenv('TOOLS_DIR')
sys.path.append(modelsDir)
sys.path.append(toolsDir)
import sae
import lstm
import cnn
import cudnnLstm
import prepareData
import fileUtils

ResDir = 'Ensembel_ResDir'
if not os.path.isdir(ResDir):
    os.makedirs(ResDir)

modelDir = 'Ensemble_ModelDir'
if not os.path.isdir(modelDir):
    os.makedirs(modelDir)


def write2file(contents, fpath):
    output = os.path.join(ResDir, fpath)
    with open(output, 'w') as f:
        f.write(contents)


def differenciate(fpath):
    modelName = ''
    fname = os.path.basename(fpath)
    if re.search('cnn', fname):
        modelName = 'cnn'
    elif re.search('sae', fname):
        modelName = 'sae'
    elif re.search('Lstm', fname):
        modelName = 'cudnnLstm'
    elif re.search('lstm', fname):
        modelName = 'cudnnLstm'
    else:
        return '', -1
    m = re.search('[0-9]', fname)
    if m:
        index = m.group(0)
    else:
        index = -1
    return modelName, int(index)


class MyOptions():
    def __init__(self, dataType, model, verbose=True, plotModel=False):
        self.dataType = dataType
        self.model = model
        self.verbose = verbose
        self.plotModel = plotModel
        self.mode = 'onlyIncoming'


class EnsembleModel():
    def __init__(self, opts):
        self.name = 'ensemble'
        self.nFold = int(opts.nFold)
        self.test_size = 0.2
        self.dataType = opts.dataType
        self.input = opts.input
        self.output = opts.output
        self.mode = opts.mode

        self.cnnModelPath = ''
        self.lstmModelPath = ''
        self.saeModelPath = ''
        self.opts = MyOptions(opts.dataType, '')
        self.weighted = opts.weightedTest

    def loadModels(self, modelDir, NUM_CLASS):
        cnnDict = defaultdict()
        lstmDict = defaultdict()
        saeDict = defaultdict()
        fList = fileUtils.genfilelist(modelDir)
        for fpath in fList:
            modelName, index = differenciate(fpath)
            if -1 == index:
                continue
            tmpDict = defaultdict()
            self.opts.model = modelName
            model, params = chooseModel(self.opts, True, NUM_CLASS=NUM_CLASS, modelPath=fpath)
            if 'cnn' == modelName:
                cnnDict[index] = [model, params]
            elif 'cudnnLstm' == modelName:
                lstmDict[index] = [model, params]
            elif 'sae' == modelName:
                saeDict[index] = [model, params]

        if not len(cnnDict.keys()) == len(saeDict.keys()) == len(lstmDict.keys()) == self.nFold:
            raise ValueError('models number for testing is not equal to nFold number: ', self.nFold)
        return cnnDict, lstmDict, saeDict

    def train_val(self, allData_raw, allLabel_raw, iter, weighted):
        if weighted:
            print('\nif weighted, further split val set\n')
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(allData_raw, allLabel_raw, test_size=0.1, shuffle=True, random_state=74)
        else:
            X_train_raw, y_train_raw = allData_raw, allLabel_raw

        modelNameList = ['sae', 'cnn', 'cudnnLstm']
        acc_dict = defaultdict()
        self.NUM_CLASS = len(set(allLabel_raw))
        model_dict = defaultdict()
        for modelName in modelNameList:
            print('now training model: {}'.format(modelName))
            self.opts.model = modelName
            model, params = chooseModel(self.opts, ifTest=False, NUM_CLASS=self.NUM_CLASS)
            X_train = prepareData.cutData(params['data_dim'], X_train_raw)
            X_train, y_train = processData(modelName, X_train, y_train_raw, self.NUM_CLASS)
            if weighted:
                X_test = prepareData.cutData(params['data_dim'], X_test_raw)
                X_test, y_test = processData(modelName, X_test, y_test_raw, self.NUM_CLASS)

            modelPath = model.train(X_train, y_train, self.NUM_CLASS)

            modelPath_prefix, modelPath_surfix = os.path.splitext(modelPath)
            newPath = modelPath_prefix + '_' + str(iter) + '_'  + self.dataType + modelPath_surfix
            print('save model {} of iter {} to {}'.format(modelName, str(iter), newPath))
            os.rename(modelPath, newPath)

            model_dict[modelName] = (model, params)
            if weighted:
                tmp_acc = model.test(X_test, y_test, self.NUM_CLASS, newPath)
                acc_dict[modelName] = tmp_acc
                print('model {} acc is: {:f}'.format(modelName, tmp_acc))
            else:
                acc_dict = {}
                print('average model, no val acc')

            model_dict[modelName] = (model, params, newPath)

        return acc_dict, model_dict

    def test(self, model_dict, acc_dict, X_test_raw, y_test_raw, i):
        NUM_CLASS = len(set(y_test_raw))
        acc_list = []

        pred_res_dict = defaultdict()
        for key in model_dict.keys():
            (model, params, modelPath) = model_dict[key]

            X_test = prepareData.cutData(params['data_dim'], X_test_raw)
            X_test, y_test = processData(key, X_test, y_test_raw, NUM_CLASS)
            model_pred = model.prediction(X_test, NUM_CLASS, modelPath)
            tmp = np.argmax(model_pred, 1)
            tmpAcc = accuracy_score(tmp, y_test_raw)
            acc_list.append(tmpAcc)
            print('{} model acc is: {:f}'.format(key, tmpAcc))

            pred_res_dict[key] = model_pred

        #assert(len(pred_res_dict.keys()) == 3)

        acc = self.merge_res(pred_res_dict, NUM_CLASS, y_test_raw, acc_dict)

        return acc

    def merge_res(self, pred_res_dict, NUM_CLASS, y_test_raw, acc_dict):
        test_num = len(y_test_raw)
        merge_res = np.zeros([test_num, NUM_CLASS])
        if {} != acc_dict:
            print('merge with weights')
            assert(pred_res_dict.keys() == acc_dict.keys())
        weightSum = sum(acc_dict.values())
        for key in pred_res_dict.keys():
            if {} != acc_dict:
                merge_res = merge_res + pred_res_dict[key] * (acc_dict[key]/weightSum)
            else:
                merge_res = merge_res + pred_res_dict[key]
        y_pred = np.argmax(merge_res, 1)
        acc = accuracy_score(y_pred, y_test_raw)

        return acc

    def run_train_test(self):
        acc_list = []
        allData, allLabel, labelMap = prepareData.loadData(self.input, 700, self.dataType, self.mode)
        NUM_CLASS = len(set(allLabel))

        for i in range(self.nFold):
            X_train, X_test, y_train, y_test = train_test_split(allData, allLabel, test_size=self.test_size, shuffle=True, random_state=i**i)

            acc_dict, model_dict = self.train_val(X_train, y_train, i, self.weighted)

            # use the value of acc_dict to control weighted or not, acc_dict == {} means no weight
            if not self.weighted:
                acc_dict = {}
            acc = self.test(model_dict, acc_dict, X_test, y_test, i)
            acc_list.append(acc)

        print('the weight used here is: ', acc_dict)
        avg_acc = mean(acc_list)
        tmp = 'acc list is: '+str(acc_list)
        tmp2 = 'average acc is: '+str(avg_acc)
        contents = '\n'.join([tmp, tmp2])
        write2file(contents, self.output)

        return acc_list, avg_acc


def chooseModel(opts, ifTest=True, NUM_CLASS=100, modelPath=''):
    if 'cnn' == opts.model:
        if 'onlyIncoming' == opts.mode:
            params = cnn.generate_default_onlyIncoming_params(opts.dataType)
        else:
            params = cnn.generate_default_whole_params(opts.dataType)
        modelObj = cnn.CNN(opts, params)
    elif 'cudnnLstm' == opts.model:
        if 'onlyIncoming' == opts.mode:
            params = cudnnLstm.generate_default_onlyIncoming_params(opts.dataType)
        else:
            params = cudnnLstm.generate_default_whole_params(opts.dataType)
        modelObj = cudnnLstm.LSTM(opts, params)
    elif 'sae' == opts.model:
        if 'onlyIncoming' == opts.mode:
            params = sae.generate_default_onlyIncoming_params(opts.dataType)
        else:
            params = sae.generate_default_whole_params(opts.dataType)
        modelObj = sae.SAE(opts, params)
    if ifTest:
        modelObj = modelObj.create_model(NUM_CLASS)
        modelObj.load_weights(modelPath)
    return modelObj, params


def reshapeData(allData, modelName):
    if 'cudnnLstm' == modelName or 'cnn' == modelName:
        allData = allData.reshape(allData.shape[0], allData.shape[1], 1)
    elif 'sae' == modelName:
        return allData
    elif 'ensemble' == modelName:
        return allData
    else:
        raise

    return allData


def processData(modelName, X_train, y_train, NUM_CLASS):
    X_train = reshapeData(X_train, modelName)
    y_train = np_utils.to_categorical(y_train, NUM_CLASS)
    return X_train, y_train


def runTest(opts):
    ensembleModel = EnsembleModel(opts)
    acc_list, acc = ensembleModel.run_train_test()
    print('acc list is:', acc_list)
    print('average acc is:', acc)


def main(opts):
    runTest(opts)


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weightedTest', action='store_true', help='weightedTest or not')
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-d', '--dataType', help='choose from onlyOrder/both')
    parser.add_argument('-n', '--nFold', default=5, help='the fold number for n fold test')
    parser.add_argument('-o', '--output', default='test_res', help='path to store results')
    parser.add_argument('-m', '--mode', help='choose to use $onlyIncoming$ or $onlyOutgoing$ or $all the data$')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-p', '--plotModel', action='store_true', help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
