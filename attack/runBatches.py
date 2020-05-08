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


def runTestOnly(modelPath, modelObj, X_test, y_test_raw, NUM_CLASS):
    y_pred_tmp = modelObj.prediction(X_test, NUM_CLASS, modelPath)
    y_pred = np.argmax(y_pred_tmp, 1)
    acc = accuracy_score(y_test_raw, y_pred)
    return acc


def doExperiment(X_train_raw, y_train_raw, testDataList, params, modelObj):
    X_train = prepareDataEnsemble.cutData(params['data_dim'], X_train_raw)
    NUM_CLASS = params['NUM_CLASS']
    X_train, y_train = ensemble.processData(modelObj.name, X_train, y_train_raw, NUM_CLASS)

    modelPath = modelObj.train(X_train, y_train, NUM_CLASS)

    acc_list = []
    for item in testDataList:
        (X_test_raw, y_test_raw) = item
        X_test = prepareDataEnsemble.cutData(params['data_dim'], X_test_raw)
        X_test, y_test = ensemble.processData(modelObj.name, X_test, y_test_raw, NUM_CLASS)
        acc = runTestOnly(modelPath, modelObj, X_test, y_test_raw, NUM_CLASS)
        acc_list.append(acc)

    return acc_list


def runTests(opts):
    modelList = ['cnn', 'sae', 'cudnnLstm']
    modelPathList = []
    report_list = []

    # load original data
    allData, allLabel, _ = prepareData.loadData(opts.input, 600, opts.dataType)
    print('finish loading original data')
    X_train_raw, X_test_raw_1, y_train_raw, y_test_raw_1 = train_test_split(allData, allLabel, test_size=0.2, shuffle=True, random_state=77)
    NUM_CLASS = len(set(y_test_raw_1))

    # load batch data
    batch1_dir = '/home/carl/work_dir/data/batches/2nd_round'
    batch2_dir = '/home/carl/work_dir/data/batches/3rd_round'
    X_test_raw_2, y_test_raw_2, _ = prepareData.loadData(batch1_dir, 600, opts.dataType)
    print('finish loading batch1 data')
    X_test_raw_3, y_test_raw_3, _ = prepareData.loadData(batch2_dir, 600, opts.dataType)
    print('finish loading batch2 data')

    testDataList = [(X_test_raw_1, y_test_raw_1), (X_test_raw_2, y_test_raw_2), (X_test_raw_3, y_test_raw_3)]


    for model in modelList:
        print('select model {} to with dataType {} to run defense testing...'.format(model, opts.dataType))
        myOpts = MyOptions(model, opts.dataType)
        params, modelObj = nFold.chooseModel(myOpts)
        params['NUM_CLASS'] = NUM_CLASS

        print('start experiment with model {}'.format(model))
        acc_list = doExperiment(X_train_raw, y_train_raw, testDataList,  params, modelObj)
        tmp = 'model {} with dataType {} and has an accuracy list {}'.format(model, opts.dataType, acc_list)
        print(tmp)
        report_list.append(tmp)

    '''
    ensemble_acc, ensemble_fp, ensemble_tp = doEnsembleTest(modelPathList, X_test_raw_d, y_test_raw_d, NUM_CLASS, opts.dataType)
    tmp = 'ensemble results with dataType {} and train with Original Data and Test with Defense data to run defense testing has an accuracy {:f}'.format(opts.dataType, ensemble_acc)
    report_list.append(tmp)
    tmp = 'ensemble false positive rate is: {:f}, true positive rate is: {:f}'.format(ensemble_fp, ensemble_tp)
    report_list.append(tmp)
    '''

    contents = '\n#########################\n'.join(report_list)
    contents = contents + '\n'
    nFold.write2file(contents, opts.output)


def main(opts):
    runTests(opts)


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='original data dir')
    parser.add_argument('-d', '--dataType', default='both', help='since we are only care about the both datatype now')
    parser.add_argument('-o', '--output', default='', help='specify the top openworld data dir')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-t', '--testOnly', default='', help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
