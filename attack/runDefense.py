#encoding=utf-8

import os
import sys
import argparse
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from statistics import mean
from collections import defaultdict
import numpy as np

if '1' == os.getenv('useGpu'):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
from keras.utils import np_utils

import sae
import lstm
import cnn
import cudnnLstm
import prepareData
import nFold
import ensemble
import prepareDataEnsemble


ResDir = nFold.ResDir
if not os.path.isdir(ResDir):
    os.makedirs(ResDir)

modelDir = nFold.modelDir
if not os.path.isdir(modelDir):
    os.makedirs(modelDir)


class MyOptions():
    def __init__(self, model, dataType, verbose=True, plotModel=False):
        self.model = model
        self.verbose = verbose
        self.plotModel = plotModel
        self.dataType = dataType


def compute_tp_and_fp(y_true, y_pred):
    tmpList = confusion_matrix(y_true=y_true, y_pred=y_pred)
    tp = tmpList[0][0] / sum(tmpList[0])
    fp = tmpList[1][0] / sum(tmpList[1])
    return tp, fp

def doEnsembleTest(modelPathList, X_test_raw, y_test_raw, NUM_CLASS, dataType):
    ensemble_pred = 0
    for modelPath in modelPathList:
        if re.search('cnn', modelPath):
            opts = MyOptions('cnn', dataType)
        elif re.search('sae', modelPath):
            opts = MyOptions('sae', dataType)
        elif re.search('Lstm', modelPath) or re.search('lstm', modelPath):
            opts = MyOptions('cudnnLstm', dataType)
        modelObj, params = ensemble.chooseModel(opts, False)
        X_test = prepareDataEnsemble.cutData(params['data_dim'], X_test_raw)
        X_test, y_test = ensemble.processData(opts.model, X_test, y_test_raw, NUM_CLASS)
        model = modelObj.create_model(NUM_CLASS)
        model.load_weights(modelPath)

        pred = model.predict(X_test)
        ensemble_pred = ensemble_pred + pred

    y_pred = np.argmax(ensemble_pred, 1)
    acc = accuracy_score(y_pred, y_test_raw)
    print('ensemble acc of defense test is: {:f}'.format(acc))
    tp, fp = compute_tp_and_fp(y_test_raw, y_pred)
    print('ensemble false positive rate is: {:f}, true positive rate is: {:f}'.format(fp, tp))
    return acc, fp, tp


def doExperiment(X_train_raw, y_train_raw, X_test_raw, y_test_raw, params, modelObj):
    X_train = prepareDataEnsemble.cutData(params['data_dim'], X_train_raw)
    X_test = prepareDataEnsemble.cutData(params['data_dim'], X_test_raw)
    NUM_CLASS = params['NUM_CLASS']
    X_train, y_train, X_test, y_test = nFold.processData(modelObj.name, X_train, y_train_raw, X_test, y_test_raw, NUM_CLASS)

    modelPath = modelObj.train(X_train, y_train, NUM_CLASS)
    y_pred_tmp = modelObj.prediction(X_test, NUM_CLASS, modelPath)
    y_pred = np.argmax(y_pred_tmp, 1)
    acc = accuracy_score(y_test_raw, y_pred)
    tp, fp = compute_tp_and_fp(y_test_raw, y_pred)
    # true postive rate and false positive rate
    return acc, fp, tp, modelPath


def runTests(opts):
    epsilon = os.path.basename(opts.defense)
    modelList = ['sae', 'cnn', 'cudnnLstm']
    #modelList = ['cnn']
    modelPathList = []
    report_list = []

    print('load original data...')
    o_allData, o_allLabel, _ = prepareData.loadData(opts.original, 600, opts.dataType)
    X_train_raw_o, X_test_raw_o, y_train_raw_o, y_test_raw_o = train_test_split(o_allData, o_allLabel, test_size=0.2, shuffle=True, random_state=77)

    print('\nload defense data...')
    d_allData, d_allLabel, _ = prepareData.loadData(opts.defense, 600, opts.dataType)
    X_train_raw_d, X_test_raw_d, y_train_raw_d, y_test_raw_d = train_test_split(d_allData, d_allLabel, test_size=0.2, shuffle=True, random_state=44)

    try:
        assert(len(set(y_test_raw_d)) == len(set(y_test_raw_o)))
    except Exception:
        print('y_test_raw len is {:d}, y_test_denfense len is {:d}'.format(len(set(y_test_raw_o)), len(set(y_test_raw_d))))

    NUM_CLASS = len(set(y_test_raw_o))
    for model in modelList:
        print('select model {} to with dataType {} and epsilon={} to run defense testing...'.format(model, opts.dataType, epsilon))
        myOpts = MyOptions(model, opts.dataType)
        params, modelObj = nFold.chooseModel(myOpts)
        params['NUM_CLASS'] = NUM_CLASS

        print('train with original data and test with defense data...')
        acc, fp, tp, modelPath = doExperiment(X_train_raw_o, y_train_raw_o, X_test_raw_d, y_test_raw_d,  params, modelObj)
        modelPathList.append(modelPath)
        tmp = 'model {} with dataType {} and epsilon={} to train with Original Data and Test with Defense data has a defense testing accuracy {:f}'.format(model, opts.dataType, epsilon, acc)
        print(tmp)
        report_list.append(tmp)
        tmp = 'model {} false positive rate is: {:f} and its true positive rate is: {:f}'.format(model, fp, tp)
        report_list.append(tmp)

    ensemble_acc, ensemble_fp, ensemble_tp = doEnsembleTest(modelPathList, X_test_raw_d, y_test_raw_d, NUM_CLASS, opts.dataType)
    tmp = 'ensemble results with dataType {} and epsilon={} to train with Original Data and Test with Defense data has a defense testing accuracy {:f}'.format(opts.dataType, epsilon, ensemble_acc)
    report_list.append(tmp)
    tmp = 'ensemble false positive rate is: {:f}, true positive rate is: {:f}'.format(ensemble_fp, ensemble_tp)
    report_list.append(tmp)

    contents = '\n#########################\n'.join(report_list)
    contents = contents + '\n'
    nFold.write2file(contents, opts.output + '_train_with_origin')

    report_list = []
    NUM_CLASS = len(set(y_test_raw_d))
    for model in modelList:
        print('select model {} to with dataType {} to run defense testing...'.format(model, opts.dataType))
        myOpts = MyOptions(model, opts.dataType)
        params, modelObj = nFold.chooseModel(myOpts)
        params['NUM_CLASS'] = NUM_CLASS

        print('train with defense data and test with defense data...')
        acc, fp, tp, modelPath = doExperiment(X_train_raw_d, y_train_raw_d, X_test_raw_d, y_test_raw_d,  params, modelObj)
        modelPathList.append(modelPath)
        tmp = 'model {} with dataType {} and epsilon={} to train with Defense Data and Test with Defense data have a defense testing accuracy {:f}'.format(model, opts.dataType, epsilon, acc)
        print(tmp)
        report_list.append(tmp)
        tmp = 'model {} false positive rate is: {:f} and its true positive rate is: {:f}'.format(model, fp, tp)
        report_list.append(tmp)

    ensemble_acc, ensemble_fp, ensemble_tp = doEnsembleTest(modelPathList, X_test_raw_d, y_test_raw_d, NUM_CLASS, opts.dataType)
    tmp = 'ensemble results with dataType {} and epsilon={} to train with Defense Data and Test with Defense data have a defense testing accuracy {:f}'.format(opts.dataType, epsilon, ensemble_acc)
    report_list.append(tmp)
    tmp = 'ensemble false positive rate is: {:f}, true positive rate is: {:f}'.format(ensemble_fp, ensemble_tp)
    report_list.append(tmp)

    contents = '\n#########################\n'.join(report_list)
    contents = contents + '\n'
    nFold.write2file(contents, opts.output + '_train_with_defense')


def main(opts):
    runTests(opts)

def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-ori', '--original', help='original data dir')
    parser.add_argument('-de', '--defense', help='defense data dir')
    parser.add_argument('-d', '--dataType', default='both', help='since we are only care about the both datatype now')
    parser.add_argument('-o', '--output', default='', help='specify the top openworld data dir')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-t', '--testOnly', default='', help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
