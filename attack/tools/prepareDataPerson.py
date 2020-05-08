#encoding=utf-8
#!/usr/bin/python

import os
import sys
import numpy as np
import pandas as pd
import re

import fileUtils
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import prepareData


def getPersonName(fpath):
    fname = os.path.basename(fpath)
    tmp = fname.split('_??_')[-1]
    prefix = os.path.splitext(tmp)[0]
    m = re.match('([a-zA-Z]*)_.*', prefix)
    if m:
        return m.group(1)
    return ''


def genFileList(droot):
    subDirs = os.listdir(droot)
    fList = []
    for subDir in subDirs:
        dpath = os.path.join(droot, subDir)
        tmpList = fileUtils.genfilelist(dpath)
        fList.extend(tmpList)
    return fList


def getPersonList(droot):
    pSet = set()
    fList = genFileList(droot)
    for fpath in fList:
        tmpName = getPersonName(fpath)
        pSet.add(tmpName)
    return list(pSet)


def loadDataDict(droot, data_dim, dataType):
    fList = genFileList(droot)
    labelMap = prepareData.getLabelMap(fList)

    personDict = defaultdict(dict)

    for fp in fList:
        if 0 == os.path.getsize(fp):
            print('skip empty file {}'.format(fp))
            continue
        personName = getPersonName(fp)
        if not personName:
            raise
        tmpData = prepareData.readFile(fp, data_dim, dataType)
        tmpLabel = prepareData.mapLabel(fp, labelMap)
        if {} == personDict[personName]:
            tmpDict = {'data':[], 'label':[]}
            tmpDict['data'].append(tmpData)
            tmpDict['label'].append(tmpLabel)
            personDict[personName] = tmpDict
        else:
            personDict[personName]['data'].append(tmpData)
            personDict[personName]['label'].append(tmpLabel)

    return personDict, labelMap


def splitDataOfDict(personDict, holdName, dataType):
    X_train, y_train = [], []
    X_test, y_test = personDict[holdName]['data'], personDict[holdName]['label']

    for key in personDict.keys():
        if key != holdName:
            X_train.extend(personDict[key]['data'])
            y_train.extend(personDict[key]['label'])

    X_train = np.array(X_train)
    y_train = np.array(y_train, dtype=np.uint8)
    X_test = np.array(X_test)
    y_test = np.array(y_test, dtype=np.uint8)
    if 'both' == dataType:
        X_train = preprocessing.scale(X_train)
        X_test = preprocessing.scale(X_test)
    return X_train, y_train, X_test, y_test
