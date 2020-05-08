#!/usr/bin/python

import os
import sys
import argparse
import re
import numpy as np
import pandas as pd

import prepareData
import prepareDataNum

import fileUtils


def loadRawData(fpath, data_dim):
    data = pd.read_csv(fpath, sep=',')
    newData = data.values
    newData = newData[:,:data_dim]
    return newData


def loadData(droot, dataType, data_dim=700):
    fMonitored = os.path.join(droot, 'monitored', 'monitored_{}.csv'.format(dataType))
    fNot = os.path.join(droot, 'unmonitored', 'not_{}.csv'.format(dataType))

    monitoredData = loadRawData(fMonitored, data_dim)
    notData = loadRawData(fNot, data_dim)

    monitoredDataLabel = np.ones(len(monitoredData), dtype=int)
    notDataLabel = np.zeros(len(notData), dtype=int)

    allData = np.concatenate((monitoredData, notData), axis=0)
    allLabel = np.concatenate((monitoredDataLabel, notDataLabel), axis=0)
    return allData, allLabel


def generateMonitoredData(opts, monitoredDir, mode):
    print('generate monitored onlyOrder data now...')
    allData, _, _ = prepareDataNum.loadData(opts.input, 800, 200, 'onlyOrder', mode)
    data = pd.DataFrame(allData)
    fname = 'monitored_onlyOrder.csv'
    fpath = os.path.join(monitoredDir, fname)
    print('data save to {}'.format(fpath))
    data.to_csv(fpath, sep=',')

    print('generate monitored both data now...')
    allData, _, _ = prepareDataNum.loadData(opts.input, 800, 200, 'both', mode)
    data = pd.DataFrame(allData)
    fname = 'monitored_both.csv'
    fpath = os.path.join(monitoredDir, fname)
    print('data save to {}'.format(fpath))
    data.to_csv(fpath, sep=',')


def generateNotMonitoredData(opts, unmonitoredDir, mode):
    print('generate unmonitored onlyOrder data now...')
    allData, _, _ = prepareData.loadData(opts.input, 800, 'onlyOrder', mode)
    data = pd.DataFrame(allData)
    fname = 'not_onlyOrder.csv'
    fpath = os.path.join(unmonitoredDir, fname)
    print('data save to {}'.format(fpath))
    data.to_csv(fpath, sep=',')

    print('generate data unmonitored both now...')
    allData, _, _ = prepareData.loadData(opts.input, 800, 'both', mode)
    data = pd.DataFrame(allData)
    fname = 'not_both.csv'
    fpath = os.path.join(unmonitoredDir, fname)
    print('data save to {}'.format(fpath))
    data.to_csv(fpath, sep=',')


def main(opts):
    OUTDIR = os.path.join(opts.output, 'OpenWorldData')
    monitoredDir = os.path.join(OUTDIR, 'monitored')
    unmonitoredDir = os.path.join(OUTDIR, 'unmonitored')
    if not os.path.isdir(monitoredDir):
        os.makedirs(monitoredDir)
    if not os.path.isdir(unmonitoredDir):
        os.makedirs(unmonitoredDir)

    if 'monitored' == opts.type:
        generateMonitoredData(opts, monitoredDir, opts.mode)
    else:
        generateNotMonitoredData(opts, unmonitoredDir, opts.mode)


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='choose from monitored/not')
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='specify the top data dir')
    parser.add_argument('-m', '--mode', help='specify the top data dir')
    parser.add_argument('-d', '--data_dim', help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
