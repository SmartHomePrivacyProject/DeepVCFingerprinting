#encoding=utf-8

import os
import sys


class Context():
    def __init__(self, dpath, data_dim, epochs1, epochs2, verbose):
        self.dpath = dpath
        self.data_dim = data_dim
        self.epochs1 = epochs1
        self.epochs2 = epochs2
        self.verbose = verbose

def createDir(dpath):
    if not os.path.isdir(dpath):
        os.makedirs(dpath)
        print('dir {} is created successfully'.format(dpath))
    else:
        print('dir {} is already exists'.format(dpath))
