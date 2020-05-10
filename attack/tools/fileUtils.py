#!/usr/bin/python

import os
import sys


def str2int(str_input):
    if str_input.find('.'):
        return int(float(str_input))
    else:
        return int(str_input)


def str2float(str_input):
    return float(str_input)


def readTxtFile(fpath, ignore):
    with open(fpath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(ignore):
                continue
            yield line


def writeTxtFile(fpath, content):
    with open(fpath, 'w') as f:
        f.write(content)


def readBinFile():
    pass


def writeBinFile():
    pass


def genfilelist(pathDir):
    fnameList = os.listdir(pathDir)
    return list(map(lambda x: os.path.join(pathDir, x), fnameList))


def getLabel(fpath):
    fname = os.path.basename(fpath)
    tmpList = fname.split('_??_')
    tmpName = tmpList[0]
    return tmpName[0:-1]


def getSubDirs(dpath):
    return os.listdir(dpath)
