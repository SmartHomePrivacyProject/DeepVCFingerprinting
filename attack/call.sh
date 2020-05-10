#!/bin/sh

export ROOT_DIR=$PWD
export TOOLS_DIR=$ROOT_DIR/tools
export MODELS_DIR=$ROOT_DIR/models


/usr/bin/nvidia-smi > /dev/null

if [ $? -eq 0 ]
then
    echo 'nvidia-smi check pass' `date`
    echo 'will run with GPU'
    export useGpu=1
else
    echo 'nvidia-smi not exists, will run with CPU'
    export useGpu=0
fi

python3 $*
