## Description
This folder constains the code for performing Voice Command Fingerprinting Attack.

### The file structure are shown below:

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
│       ├── sae-keras.py  
│       └── search_space.json  
├── tools  
│   ├── common_use.py  
│   ├── fileUtils.py  
│   ├── prepareData4OpenWorld.py  
│   ├── prepareDataNum.py  
│   ├── prepareDataPerson.py  
│   ├── prepareData.py  
│   └── writeRes.py  
├── README.md  
└── requirements.txt  


### The description of the files as shown below:

All DNN model definition file are in ./models  
* In ./tools, those code are used for help preprocessing the data and other support functionalities
* In ./expriments, those code are for running different experiments
* In ./searchParams, those code are for searching parameters for DNNs
* File call.sh is used to facility you running with those experiments

## Usage
### If you want to reproduce the experiments results, you can call command:
* nfold test
    ./call.sh experiments/nfold.py 
* ensemble test
    ./call.sh experiments/ensemble.py
* Defense test
    ./call.sh experiments/runDefense.py 
* Open-world test
    ./call.sh experiments/runOpenWorld.py 
    
### if you want to search parameters for the neural network, you can call command:  
* nnictl create --config the_config_file_for_the_DNN_you_choose
