#!/usr/bin/env python

import ipaddress
import sys
from pathlib import Path
import pandas as pd
import csv
from scapy.all import *
# from scapy_ssl_tls.ssl_tls import TLS


def pcap_converter(pcap_path, echo_ip, burst_ranges, dst):
    """
    Take the pcap files, convert to a list, break list into trace files based of number of queries,
    save sublists to csv files after extracting relevant packet information.
    :param pcap_path:
    :param echo_ip: IPv4 address of the Amazon Echo
    :param burst_ranges: list of tuples in the form (f_p, l_p) where f_p is the first packet in the burst and l_p is the
                         last
    :return: nothing. saves pandas dataframes in csv format
    """
    # csv_path = 'csv/' + dst + '/'
    csv_path = 'csv/'
    pf = Path(pcap_path)
    trace_name = pf.name[0:-5]
    packets = rdpcap(pcap_path)

    p_list = packets[burst_ranges[0]:burst_ranges[1]]

    # print(init_time)
    # echo_df = pd.DataFrame(columns=['time', 'size', 'direction','protocol'])
    echo_df = pd.DataFrame(columns=['time', 'size', 'direction'])
    p_list.reverse()
    echo_packets = []
    for p in p_list:
        # p.show()
        try:
            p[IP].src
            p[TCP]
        except:
            continue
        if p[IP].src.strip() == echo_ip.strip() or p[IP].dst.strip() == echo_ip.strip():
            if p[TCP] and p.len < 60: #filter ACk packets
                continue
            else:
                echo_packets.append(p)
    p_list = echo_packets

    init_time = p_list[-1].time
    # print("Number of packets in csv files: {}" .format(len(p_list)))
    for p in p_list:
        # 1 if echo is src, -1 if destination
        if p[IP].src == echo_ip:
            p[IP].src = 1
        elif p[IP].dst == echo_ip:
            p[IP].src = -1
        else:
            p[IP].src = 0

        # Update the df with correct index
        # echo_df.loc[-1] = [p.time - init_time, p.len, p[IP].src, p[IP].proto]
        echo_df.loc[-1] = [p.time - init_time, p.len, p[IP].src]
        echo_df.index = echo_df.index + 1

        # Sort, so list starts in non-reverse order, save to csv
        echo_df = echo_df.sort_index()
        echo_df.to_csv(csv_path + trace_name + ".csv", index=False)


def burst_detector_short(packet_path, echo_ip):
    """
    Detect Burst Patterns in pcap files, getting rid of need to manually enter first and last packet indices associated
    with queries
    Short version: For shorter packet burst detection
    :return:
    """

    interval = 1  # Amount of time between packets that constitutes a burst
    packets = rdpcap(packet_path)  # Read packet file
    c = 0  # Counter that stores number of packets in burst
    # burst_in_progress = False
    echo_packets = []

    # only echo packets
    for p in packets:
        try:
            p[IP].src
            p[TCP]
        except:
            continue
        if p[IP].src.strip() == echo_ip.strip() or p[IP].dst.strip() == echo_ip.strip():
            echo_packets.append(p)

    packets = echo_packets

    for i, p in enumerate(packets):
        if i == 0:  # fist
            continue
        if i == len(packets) - 1:  # last
            continue
        prev_packet = packets[i - 1]
        next_packet = packets[i + 1]
        c+=1

        if ((abs(p.time - prev_packet.time) < interval) and (abs(next_packet.time - p.time) > interval)):
            # End of the current trace
            if i < int(0.75 * len(packets)):
                continue
            else:
                c += 1
                break
    return c


def main(argv):

    pcap_path = argv[0]
    echo_ip = argv[1]
    dst = argv[2]

    try:
        ipaddress.ip_address(echo_ip)
    except ValueError:
        print("ValueError: You did not enter a valid IPv4 address")

    pf = Path(pcap_path)
    if not pf.is_file():
        raise FileNotFoundError("No file exists at specified path" + pcap_path)
    # print('pcap file ' + pf.name + ' loaded.')
    print('Running burst detection...')

    end_index = burst_detector_short(pcap_path, echo_ip)
    ranges = (1, end_index);
    # print('packet number ranges of {}'.format(ranges))
    # ranges = burst_detector_short(pcap_path, echo_ip)
    # print('Burst detection finished.')
    # print('There are {} bursts with packet number ranges of {}'.format(len(ranges), ranges))
    # print('Using ranges to convert pcap files to CSV trace files...')

    if end_index > 50:
        pcap_converter(pcap_path, echo_ip, ranges, dst)
        print(dst + ' ' + pf.name)

if __name__ == "__main__":
    main(sys.argv[1:])
    


