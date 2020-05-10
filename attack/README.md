## Description
This folder constains the code for performing Voice Command Fingerprinting Attack.

### The file structure are shown belwo:

attack
├── call.sh
├── experiments
│   ├── ensemble.py
│   ├── nFold.py
│   ├── runDefense.py
│   └── runOpenWorld.py
├── models
│   ├── cnn.py
│   ├── cudnnLstm.py
│   ├── lstm.py
│   └── sae.py
├── paramsSearch
│   ├── cnn-keras
│   │   ├── cnn-keras.py
│   │   ├── config_pai.yml
│   │   ├── config.yml
│   │   └── search_space.json
│   ├── lstm-keras
│   │   ├── config_pai.yml
│   │   ├── config.yml
│   │   ├── lstm-keras.py
│   │   └── search_space.json
│   └── sae-keras
│       ├── config_pai.yml
│       ├── config.yml
│       ├── lstm-keras.py
│       ├── modelDir
│       │   └── lstm_weights_best.hdf5
│       ├── sae-keras.py
│       └── search_space.json
├── README.md
├── requirements.txt
└── tools
    ├── common_use.py
    ├── fileUtils.py
    ├── prepareData4OpenWorld.py
    ├── prepareDataNum.py
    ├── prepareDataPerson.py
    ├── prepareData.py
    └── writeRes.py  

### The description of the files as shown below:

All DNN model definition file are in ./models
In ./tools, those code are used for help preprocessing the data and other support functionalities
In ./expriments, those code are for running different experiments
In ./searchParams, those code are for searching parameters for DNNs

## Usage
If you want to reproduce the experiments results, you can call command:
* nfold test

* ensemble test

* Defense test

* Open-world test


if you want to search parameters for the neural network, you can call command:
nnictl create --config the_config_file_for_the_DNN_you_choose
