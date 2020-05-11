#!/usr/bin/python

import os
import sys
import logging


def getDirs():
    if not os.path.isdir('results'):
        os.makedirs('results')
    LOG = logging.getLogger('results/sae_results')

    modelDir = 'model_parame_search_dir'
    if not os.path.isdir(modelDir):
        os.makedirs(modelDir)
    return LOG, modelDir
