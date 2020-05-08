#encoding=utf-8
#!/usr/bin/python

import os
import sys
import numpy as np
import pandas as pd
import re

import mytools.fileUtils as fileUtils
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def getLabel_old(fpath):
    #need to negotiate the file name pattern first
    pattern = '([a-zA-Z\']*[a-zA-Z_]+)_[0-9].*'
    fname = os.path.basename(fpath)
    m = re.match(pattern, fname)
    if m:
        return m.group(1)
    else:
        return os.path.basename(fpath)

def getLabel(fpath):
    fname = os.path.basename(fpath)
    tmpList = fname.split('_??_')
    tmpName = tmpList[0]
    return tmpName

def getLabelMap(fnameList):
    cNameDict = defaultdict(int)
    count = 0
    for fname in fnameList:
        cname = getLabel(fname)
        if cname in cNameDict.keys():
            continue
        else:
            cNameDict[cname] = count
            count += 1
    return cNameDict

def mapLabel(fpath, labelMap):
    fname = os.path.basename(fpath)
    fname = getLabel(fname)
    return labelMap[fname]

def padData(dataList, expDim):
    len_dataList = len(dataList)
    #new_dataList = dataList + [0 for i in range(expDim - len_dataList)]
    dataList.extend([0]*(expDim-len_dataList))
    return dataList

def readFile_old(fpath, data_dim, dataType):
    tmpList = []
    count = 0
    for line in fileUtils.readTxtFile(fpath, 'time'):
        tmp = line.split(',')
        if 'both' == dataType:
            tmp_multi = fileUtils.str2int(tmp[-1]) * fileUtils.str2int(tmp[-2])
        else:
            tmp_multi = fileUtils.str2float(tmp[-1])
        tmpList.append(tmp_multi)
        count += 1
        if count >= data_dim:
            break
    if count < data_dim:
        tmpList = padData(tmpList, data_dim)

    allData = np.array(tmpList, dtype=float)
    return allData


def dataFiler(dataList, mode):
    tmp = []
    for elem in dataList:
        if mode == 'onlyIncoming':
            if elem < 0:
                tmp.append(elem)
        elif mode == 'onlyOutgoing':
            if elem > 0:
                tmp.append(elem)
        elif mode == 'both':
            return dataList
        else:
            raise ValueError('wrong input')
    return tmp


def readFile(fpath, data_dim, dataType, mode):
    tmpList = []
    df = pd.read_csv(fpath, sep=',', skiprows=0)
    if 'both' == dataType:
        tmp_size = df['size'].values
        tmp_direction = df['direction'].values
        tmp_multi = tmp_size * tmp_direction
    else:
        tmp_multi = df['direction'].values

    tmpList = tmp_multi[:data_dim]

    tmpList = dataFiler(tmpList, mode)

    tmpList = padData(list(tmpList), data_dim)

    allData = np.array(tmpList, dtype=float)
    return allData


def genFileList(droot):
    subDirs = os.listdir(droot)
    fList = []
    for subDir in subDirs:
        dpath = os.path.join(droot, subDir)
        tmpList = fileUtils.genfilelist(dpath)
        fList.extend(tmpList)
    return fList


def loadData(droot, data_dim, dataType, mode='both'):
    fList = genFileList(droot)
    tmpDataList = []
    tmpLabelList = []
    labelMap = getLabelMap(fList)
    count = 1
    totalNum = len(fList)
    for fp in fList:
        if 0 == os.path.getsize(fp):
            print('skip empty file {}'.format(fp))
            continue
        tmpData = readFile(fp, data_dim, dataType, mode)
        tmpDataList.append(tmpData)
        tmpLabel = mapLabel(fp, labelMap)
        tmpLabelList.append(tmpLabel)
        print("\r read {}/{} files".format(count, totalNum), end="")
        count = count + 1

    allData = np.array(tmpDataList)
    allLabel = np.array(tmpLabelList, dtype=np.uint8)

    if 'both' == dataType:
        # 0 mean 1 var
        allData = preprocessing.scale(allData)
    return allData, allLabel, labelMap


def loadTrainAndTestData(dpath, data_dim, dataType, mode):
    x, y, labelMap = loadData(dpath, data_dim, dataType, mode)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        shuffle=True, random_state=42)
    return X_train, y_train, X_test, y_test, labelMap


def cutData(data_dim, allData):
    newData = allData[:,:data_dim]
    if not data_dim == len(newData[0]):
        raise
    return newData
