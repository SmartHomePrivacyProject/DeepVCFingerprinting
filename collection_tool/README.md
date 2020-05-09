# Speaker Data Collection Platform
Automate Smart Speaker Data Collection.  

<img src=echo.jpg />.

This platform allows for automatic generation and collection of network traffic data from smart assistant-enabled speakers like the Amazon Echo and the Google Home.  Audio files are generated corresponding to queries (like 'Alexa, what is the weather today?').  These files are played from a regular speaker initiating an interaction between the speaker and the smart speaker.  At the same time a network sniffer is started on the AP the smart speaker is connected to.  The network traffic associated with the interaction is saved to a pcap file.  

## Setup

In order to set up a speaker data collection platform the following items are required:

- Smart Speaker (Amazon Echo or Google Home)
- Normal Speaker
- Raspberry Pi setup as Wireless Access Point (I have written instructions on how to do this, which can be found here, https://medium.com/think-secure/using-a-raspberry-pi-as-a-wifi-access-point-to-capture-amazon-echo-network-traffic-8c66433a388a

The normal speaker should be powered on and connected to the Raspberry Pi AP, and the Smart Speaker needs to be connected to the Raspberry Pi AP's WiFi network.

## Usage

In addition to the physical setup described above, you need to have a csv file containing the interaction time for each query (i.e., the query 'Alexa, what is the weather today?' takes 30 seconds).  

### Generating Audio Files 

Use the generate_audio_queries.py script to produce mp3 files that contain the queries the speaker will play.  By default the wake work used is "Alexa, " and the voice used is the default Google text-to-speech US-English Female voice.

```bash
python3 generate_audio_queries.py data/queries.csv 
```

### Capturing Network Traffic Data of Smart Speaker Interactions

Use the speaker_collect.py script to start the platform. You need to specify the CSV File that contains the columns Query and Time, corresponding to the query and interaction time, the  directory containing the audio query files, the device's IP address, and the number of captures to collect for each query.

```bash
sudo python3 speaker_collect.py data/file.csv voice_queries/ 192.192.192.192 100
```



