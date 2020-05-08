#!/usr/bin/python

import os
import sys
import argparse
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from statistics import mean
from collections import defaultdict

if '1' == os.getenv('useGpu'):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
from keras.utils import np_utils

import sae
import lstm
import cnn
import cudnnLstm
import ensemble
import prepareData
import prepareDataNum
import prepareDataPerson


ResDir = 'res_out_dir'
if not os.path.isdir(ResDir):
    os.makedirs(ResDir)

modelDir = 'modelDir'
if not os.path.isdir(modelDir):
    os.makedirs(modelDir)


def write2file(contents, fpath):
    output = os.path.join(ResDir, fpath)
    with open(output, 'w') as f:
        f.write(contents)


def chooseModel(opts):
    print('selecting model to test...')
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

    return params, modelObj


def reshapeData(allData, modelName):
    if 'cnn' == modelName:
        allData = np.expand_dims(allData, axis=2)
    elif 'lstm' == modelName:
        allData = allData.reshape(allData.shape[0], allData.shape[1], 1)
    elif 'cudnnLstm' == modelName:
        allData = allData.reshape(allData.shape[0], allData.shape[1], 1)
    elif 'sae' == modelName:
        return allData
    elif 'ensemble' == modelName:
        return allData
    else:
        raise ValueError('model name {} is not defined'.format(modelName))

    return allData


def processData(modelName, X_train, y_train, X_test, y_test, NUM_CLASS):
    X_train = reshapeData(X_train, modelName)
    X_test = reshapeData(X_test, modelName)
    y_train = np_utils.to_categorical(y_train, NUM_CLASS)
    y_test = np_utils.to_categorical(y_test, NUM_CLASS)
    return X_train, y_train, X_test, y_test


def doExperiment(opts, params, modelObj, X_train, y_train, X_test, y_test, count=0):
    print('start train model {}'.format(modelObj.name))
    trainStart = time.time()
    modelPath = modelObj.train(X_train, y_train, params['NUM_CLASS'])
    trainEnd = time.time()

    print('start test model...')
    testStart = time.time()
    acc = modelObj.test(X_test, y_test, params['NUM_CLASS'], modelPath)
    testEnd = time.time()

    trainTime = trainEnd - trainStart
    testTime = (testEnd - testStart)/len(y_test)

    modelPath_prefix, modelPath_surfix = os.path.splitext(modelPath)
    newPath = modelPath_prefix + '_' + str(count) + '_' + opts.dataType + modelPath_surfix
    os.rename(modelPath, newPath)

    return acc, trainTime, testTime


def runTestOnce(opts, allData, allLabel, params):
    count = 1
    acc_list = []
    trainTimeList = []
    testTimeList = []

    params, modelObj = chooseModel(opts)
    nFoldNum = int(opts.nFold)

    for i in range(nFoldNum):
        X_train, X_test, y_train, y_test = train_test_split(allData, allLabel, test_size=0.2, shuffle=True, random_state=i+44)
        print('start {} round of experiment...'.format(count))
        count = count + 1

        params['NUM_CLASS'] = len(set(y_test))
        X_train, y_train, X_test, y_test = processData(opts.model, X_train, y_train, X_test, y_test, params['NUM_CLASS'])
        acc, trainTime, testTime = doExperiment(opts, params, modelObj, X_train, y_train, X_test, y_test, i)

        acc_list.append(acc)
        trainTimeList.append(trainTime)
        testTimeList.append(testTime)

    return acc_list, trainTimeList, testTimeList


def resReport(modelName, acc_list, trainTimeList, testTimeList, sampleNum=0):
    avg_acc = mean(acc_list)
    avg_train_time = mean(trainTimeList)
    avg_test_time = mean(testTimeList)
    resultList = []
    if sampleNum:
        tmp = '{}, {}'.format(sampleNum, avg_acc)
        resultList.append(tmp)
    tmp = 'prediction with method {} has a accuracy list: '.format(modelName) + str(acc_list)
    resultList.append(tmp)
    tmp = 'prediction with method {} has a accuracy of: {}'.format(modelName, avg_acc)
    resultList.append(tmp)
    tmp = 'average training time is: {}, average testing time is: {}'.format(str(avg_train_time), str(avg_test_time))
    resultList.append(tmp)

    contents = '\n'.join(resultList)
    contents = contents + '\n'
    return contents


def runNumTest(opts):
    sampleNumList = list(range(100, 1400, 100))
    contentList = []
    params, modelObj = chooseModel(opts)

    for sampleNum in sampleNumList:
        print('extracting data...')
        allData, allLabel, labelMap = prepareDataNum.loadAndVerifyData(opts.input, params['data_dim'], sampleNum, opts.dataType)
        X_train, X_test, y_train, y_test = train_test_split(allData, allLabel, test_size=0.2, shuffle=True, random_state=47)
        params['NUM_CLASS'] = len(set(y_test))
        X_train, y_train, X_test, y_test = processData(opts.model, X_train, y_train, X_test, y_test, params['NUM_CLASS'])
        acc, trainTime, testTime = doExperiment(opts, params, modelObj, X_train, y_train, X_test, y_test)

        contents = 'model {} with dataType {} and sample Number {:d} has acc: {:f}'.format(opts.model, opts.dataType, sampleNum, acc)
        contentList.append(contents)

    allContents = '\n###############\n'.join(contentList)
    print(allContents)
    write2file(allContents, opts.output)


def runPersonTest(opts):
    personList = prepareDataPerson.getPersonList(opts.input)
    tmpList = []
    PARAMS, modelObj = chooseModel(opts)

    acc_list = []

    print('extracting data...')
    personDict, labelMap = prepareDataPerson.loadDataDict(opts.input, PARAMS['data_dim'], opts.dataType)
    print('now we are run person test, we have: ', str(personList))
    for person in personList:
        print('now we are test with {} and train with the rest'.format(person))
        X_train, y_train, X_test, y_test = prepareDataPerson.splitDataOfDict(personDict, person, opts.dataType)
        PARAMS['NUM_CLASS'] = len(set(y_test))
        X_train, y_train, X_test, y_test = processData(opts.model, X_train, y_train, X_test, y_test, PARAMS['NUM_CLASS'])

        acc, trainTime, testTime = doExperiment(opts, PARAMS, modelObj, X_train, y_train, X_test, y_test)
        tmp = 'test on {} with model {} and dataType {} has a acc {:f}'.format(person, opts.model, opts.dataType, acc)
        acc_list.append(acc)

    print(acc_list)

    contents = '\n'.join(acc_list)
    contents = contents + '\n'
    print(contents)
    write2file(contents, opts.output)


def runNormalTest(opts):
    PARAMS, modelObj = chooseModel(opts)
    if 'ensemble' == opts.model:
        ensembleModel = ensemble.EnsembleModel(opts)
        acc_list, acc = ensembleModel.run()
        print(acc_list)
        print(acc)
    else:
        allData, allLabel, labelMap = prepareData.loadData(opts.input, PARAMS['data_dim'], opts.dataType, opts.mode)
        acc_list, trainTimeList, testTimeList = runTestOnce(opts, allData, allLabel, PARAMS)
        contents = resReport(opts.model, acc_list, trainTimeList, testTimeList)
        print(contents)
        write2file(contents, opts.output)


'''
def runNormalTest(opts, PARAMS):
    params = sae.generate_default_params()
    allData, allLabel, labelMap = prepareData.loadData(opts.input, params['data_dim'], opts.dataType)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(allData, allLabel):
        X_train, X_test = allData[train_index], allData[test_index]
        y_train, y_test = allLabel[train_index], allLabel[test_index]
        params['NUM_CLASS'] = len(set(y_test))
        y_train = np_utils.to_categorical(y_train, params['NUM_CLASS'])
        y_test = np_utils.to_categorical(y_test, params['NUM_CLASS'])

        modelObj = sae.SAE(opts, params)
        print('start train model {}'.format(modelObj.name))
        modelPath = modelObj.train(X_train, y_train, params['NUM_CLASS'])

        print('start test model...')
        acc = modelObj.test(X_test, y_test, params['NUM_CLASS'], modelPath)
        print('at last: ', acc)
'''

def main(opts):
    print('selecting model {} with dataType {}'.format(opts.model, opts.dataType))

    if 'num' == opts.test:
        runNumTest(opts)
    elif 'person' == opts.test:
        runPersonTest(opts)
    elif 'normal' == opts.test:
        runNormalTest(opts)
    else:
        raise


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='choose from cnn/sae/lstm/cudnnLstm/ensemble')
    parser.add_argument('test', help='choose from num/person/normal')
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-d', '--dataType', help='choose from onlyOrder/both')
    parser.add_argument('-n', '--nFold', default=5, help='choose from onlyOrder/both')
    parser.add_argument('-o', '--output', default='test_res', help='path to store results')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-m', '--mode', help='')
    parser.add_argument('-p', '--plotModel', action='store_true', help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
