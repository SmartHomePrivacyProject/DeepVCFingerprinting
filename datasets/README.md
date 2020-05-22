## Usage
This folder contains the link of our datasets (Amazon_Echo_Dataset) and (Google_Home_Dataset) stored in OneDrive.

Amazon_Echo_Dataset has 100 classes for closed-world setting and another 100 classes for open-world setting. Each class in the closed-world setting has 1500 traffic traces (i.e., repeating a voice command 1500 times); each class in the open-world setting has 200 traffic traces. 

A list of the classes (i.e., voice commands) can be found in ./additional_info 

`Amazon_Echo_Dataset/traffic_alexa` includes both the original pcap files captured by tcpdump and also the processed csv files (which only includes timestamp, size, the direction of each packet). 

Invalid traces (e.g., the cases where a voice command was played corrrectly but there was no response from a smart speaker) were removed when we transfered the data from pcap files to csv files. The number of invalied traces was less than 1% of the entire data. More information can be found in our paper. 

The transfer from pcap files to csv files were done by using the code from `./preprocess`. 

**To run our code, please use the csv files directly.** The timestamp information was not used in our attack code, the numeric format uses both size and direction and the binary format uses direction only. 

`Amazon_Echo_Dataset/voice_commands_automated_alexa` includes the voice recordings we used for data collection. 

Goolge_Home_Dataset has 100 classes for closed-world setting. Each class has 1500 traffic traces. 

`Google_Home_Dataset/traffic_alexa` includes the original pcap files captured by tupdump and also the csv files are provided (which only includes timestamp, size, the direction of each packet). 

`Google_Home_Dataset/voice_commands_automated_google` includes the voice recordings we used for data collection. 

More detailed information about the datasets can be found in our WiSec20 paper. We made our dataset publicly avaliable. The data should be used for reasearch purpose only. 

Amazon_Echo_Dataset: https://mailuc-my.sharepoint.com/:f:/g/personal/wang2ba_ucmail_uc_edu/EqtkhDKpukxFmDgpJpQsJjcBakmkxoTnaq3PYYLK1_97Bg?e=Igvejc

Google_Home_Dataset: https://mailuc-my.sharepoint.com/:f:/g/personal/wang2ba_ucmail_uc_edu/EuNkaOqv6shIt8QEuudLkyMByHHwN8ai-dWr07DeKe7qcw?e=EB9Qzu
