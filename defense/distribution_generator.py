from __future__ import division
import csv
import sys
import argparse
from pathlib import Path
import pandas as pd
import selfUtils as su


def size_count(packets):
    outgoing = []
    incoming = []
    for p in packets:
        if p[2] == -1:
            incoming.append(p)
        else:
            outgoing.append(p)

    try:
        out_size_list = su.csv_numpy('stats/adapt_out_distribution_size.csv')
    except IOError:
        with open('stats/adapt_out_distribution_size.csv','a') as build:
            writer = csv.writer(build)
            writer.writerow(['size','count'])
        out_size_list = su.csv_numpy('stats/adapt_out_distribution_size.csv')
    try:
        in_size_list = su.csv_numpy('stats/adapt_in_distribution_size.csv')
    except IOError:
        with open('stats/adapt_in_distribution_size.csv', 'a') as build:
            writer = csv.writer(build)
            writer.writerow(['size', 'count'])
        in_size_list = su.csv_numpy('stats/adapt_in_distribution_size.csv')

    out_exist = False
    for p in outgoing:
        for s in out_size_list:
            if p[1] == s[0]:
                s[1] = s[1] + 1
                out_exist = True
                break
        if not out_exist:
            new_out_size = [p[1], 1]
            out_size_list.append(new_out_size)
        out_exist = False

    db_df = pd.DataFrame(out_size_list, columns=['size', 'count'])
    db_df.to_csv('stats/adapt_out_distribution_size.csv',index=False)

    in_exist = False
    for p in incoming:
        for s in in_size_list:
            if p[1] == s[0]:
                s[1] = s[1] + 1
                in_exist = True
                break
        if not in_exist:
            new_in_size = [p[1], 1]
            in_size_list.append(new_in_size)
        in_exist = False
    db_df = pd.DataFrame(in_size_list, columns=['size', 'count'])
    db_df.to_csv('stats/adapt_in_distribution_size.csv',index=False)


def interval_count(packets):

    init_out_list = [[0.00001, 0], [0.00005, 0], [0.0001, 0], [0.0005, 0], [0.001, 0],
                     [0.003, 0], [0.005, 0], [0.01, 0], [0.012, 0], [0.014, 0],
                     [0.016, 0], [0.018, 0], [0.02, 0], [0.025, 0], [0.03, 0],
                     [0.05, 0], [0.1, 0], [0.5, 0], [1, 0], [100000, 0]]

    init_in_list = [[0.00001, 0], [0.0001, 0], [0.00013, 0],[0.00015, 0],[0.00017, 0],
                    [0.0002, 0], [0.00025, 0], [0.0003, 0], [0.0005, 0], [0.001, 0],
                    [0.005, 0], [0.01, 0], [0.03, 0], [0.05, 0], [0.07, 0],
                    [0.1, 0], [0.5, 0], [1.0, 0], [2.0, 0], [100000, 0]]

    outgoing = []
    incoming = []
    for p in packets:
        if p[2] == -1:
            incoming.append(p)
        else:
            outgoing.append(p)

    try:
        out_interval_list = su.csv_numpy('stats/adapt_out_distribution_interval.csv')
    except IOError:
        with open('stats/adapt_out_distribution_interval.csv','a') as build:
            writer = csv.writer(build)
            writer.writerow(['interval', 'count'])
            for p in init_out_list:
                writer.writerow(p)
        out_interval_list = su.csv_numpy('stats/adapt_out_distribution_interval.csv')

    try:
        in_interval_list = su.csv_numpy('stats/adapt_in_distribution_interval.csv')
    except IOError:
        with open('stats/adapt_in_distribution_interval.csv','a') as build:
            writer = csv.writer(build)
            writer.writerow(['interval', 'count'])
            for p in init_in_list:
                writer.writerow(p)
        in_interval_list = su.csv_numpy('stats/adapt_in_distribution_interval.csv')

    for i, p in enumerate(outgoing):
        if i==0:
            continue
        out_interval = p[0] - outgoing[i-1][0]
        for k in out_interval_list:
            if out_interval <= k[0]:
                k[1]+=1
                break
    db_df = pd.DataFrame(out_interval_list,columns=['interval','count'])
    db_df.to_csv('stats/adapt_out_distribution_interval.csv',index=False)

    for i, p in enumerate(incoming):
        if i==0:
            continue
        in_interval = p[0] - incoming[i-1][0]
        for k in in_interval_list:
            if in_interval <= k[0]:
                k[1]+=1
                break
    db_df = pd.DataFrame(in_interval_list,columns=['interval','count'])
    db_df.to_csv('stats/adapt_in_distribution_interval.csv',index=False)


def main(opts):
    csv_path = opts.csvPath

    packets = su.csv_numpy(csv_path)
    pf = Path(csv_path)
    trace_name = pf.name[0:-4]
    size_count(packets)
    interval_count(packets)
    print(csv_path + 'is finished')


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-pc','--csvPath', help='path to read csv files')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
