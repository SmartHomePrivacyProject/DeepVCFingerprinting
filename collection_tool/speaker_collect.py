#!/usr/bin/env python3

import argparse
import csv
import subprocess
import time
from os import listdir
from os.path import isfile, join


def driver(query_path : str, q_time : int, q_name : str, device_ip, iterations : int, start_index: int):
    """
    Perform a tcpdump capture during an interaction with a smart speaker.  Use a list of audio files containing the query
    to initiate the interaction. At the same time an tcpdump session is started, which times out after the given q_time.
    Saves the data in pcap files and repeats for the number of iterations passed in.
    
    query_path: str. Path of directory containing query files
    q_time: int. Length of interaction time for each query
    q_name: str. Title of the query
    iterations: int.  Number of times to repeat capturing data for the query
    start_index : int. Start index of the traffic trace
    """
    for i in range(iterations):
        ## added by Boyang
        ## start_index = 200
        j = i + start_index
       
        # Sets the default inteface
        interface = 'br0'
        
        ## disabled by Boyang
        ## q_time = q_time + 1
        
        print('Starting capture...')
        
        ## change i to j 
        capture = subprocess.Popen("sudo timeout " + str(q_time) + " tcpdump -i " + str(interface) + " -n host " + str(
            device_ip) + " -w " + "/home/pi/speaker-crawler/output/" + str(q_name) + str(
            j) + "_" + ".pcap", shell=True)
        time.sleep(1)
        print('Playing audio query...')
        ## play_audio = subprocess.Popen(["mpg321", query_path])
        play_audio = subprocess.Popen(["mplayer", query_path])
        capture.wait()
        print('Interaction captured')
        
        ## change i to j 
        print("Capture File Saved, iter=" + str(j))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('csv',
                        help='A CSV File that contains the columns Query and Time, corresponding to the query and '
                             'interaction time')
    parser.add_argument('mp3_fp', help='The directory containing the audio query files')
    parser.add_argument('device_ip', help='The IP Address of the device')
    parser.add_argument('iter', help='Number of capture files to generate for each query')
    
    ## added by Boyang
    parser.add_argument('start', help='Start index of traffic trace')
    
    args = parser.parse_args()

    with open(args.csv) as csv_file:
        q_names = []
        q_times = []
        queries = csv.DictReader(csv_file)
        for row in queries:
            q_names.append(row["Query"])
            q_times.append(row["Time"])

    device_ip = args.device_ip
    query_directory = args.mp3_fp

    q_files = [f for f in listdir(query_directory) if isfile(join(query_directory, f))]
    q_files = sorted(q_files, key=lambda s: s.casefold())
    n_iter = int(args.iter)
    
    ## added by Boyang
    n_start_index = int(args.start)
    
    ## add n_start_index 
    for idx, q in enumerate(q_files):
        driver(query_directory + q_files[idx], float(q_times[idx]), q_names[idx].replace(" ", "_") + '_', device_ip,
               n_iter, n_start_index)
        # print(q, query_directory + q_files[idx], float(q_times[idx]), q_names[idx].replace(" ", "_") + '_', device_ip)
