from __future__ import division
import sys
import argparse
import random
from pathlib import Path
import pandas as pd
import selfUtils as su
import csv
from math import log
from scipy.stats import laplace
import queue
import os


def calculate_ratio(packets):
    s = r = 0
    for p in packets:
        s += p[1]
    for i,p in enumerate(packets):
        if i == 0:
            r = p[1] / s
            p.append(r)
            continue
        r = (p[1] / s) + packets[i - 1][2]
        p.append(r)


def sample_from_distribution(size_list, interval_list):
    size = interval = 0

    a = random.randint(0, 99)/100
    b = random.randint(0, 99)/100
    for p in size_list:
        if a <= p[2]:
            size = p[0]
            break
    for p in interval_list:
        if b <= p[2]:
            interval = p[0]
            break
    return size, interval


def extension(time):
    for i in range(10000):
        if time * 1000 <= pow(2, i):
            a = pow(2, i)/1000
            return a
        else:
            continue


def lap(packets, lap_list, eps):
    
    g = 0
    r = 0
    num = -1
    i = len(lap_list)

    g = su.cal_g(i)
    if i == 1 or i == su.cal_d(i):        
        r = int(laplace.rvs(0, 1/eps))
    else:
        num = int(log(i, 2))
        r = int(laplace.rvs(0, num/eps))
    x = lap_list[g][1] + (packets[i][1] - packets[g][1]) + r
    # print(g, i)
    if x > 1500:
        x = 1500
    if x < 0:
        x = 0

    n = [packets[-1][0], x, packets[-1][2]]
    return n, x - packets[-1][1]


def extend_trace_lap(packets, lap_list, size_list, interval_list, end_time, buffer_q, buffer_p, proc_q, eps):

    start_time = packets[-1][0]
    ap_overhead = 0
    lap_overhead = 0
    direction = packets[-1][2]
    dummy_n = real_n = 0
    while True:
        (size, interval) = sample_from_distribution(size_list, interval_list)
        if start_time + interval <= end_time:
            
            dummy = [packets[-1][0] + interval, size, direction, 'extended_dummy']
            packets.append(dummy)

            (lap_p, diff) = lap(packets, lap_list, eps)
            if lap_p[1] == 0:
                pass
            else:
                lap_p.append('extended_dummy')
                lap_list.append(lap_p)

                if not buffer_q.empty() or buffer_p:
                    if buffer_p:
                        pass
                    else:
                        buffer_p = buffer_q.get()
                    size_d = lap_p[1]
                    while size_d > buffer_p[2]:
                        dummy_n += 1
                        proc_q.put(buffer_p + [lap_p[0], len(lap_list)-2] + [dummy_n, real_n])
                        size_d = size_d - buffer_p[2]
                        dummy_n = real_n = 0
                        try:
                            buffer_p = buffer_q.get(False)
                        except queue.Empty:
                            buffer_p = []
                            lap_overhead = lap_overhead + size_d
                            break
                    if buffer_p:
                        buffer_p[2] = buffer_p[2] - abs(size_d)
                        dummy_n += 1
                else:
                    lap_overhead += lap_p[1]

            ap_overhead = size + ap_overhead

            start_time = packets[-1][0] + interval
        else:
            break
    return ap_overhead, lap_overhead, proc_q


def fill_gap_lap(ori_packets, size_list, interval_list, eps):
    unfinished = False
    end_time = 100.0
    ap_overhead = 0.0
    lap_overhead = 0.0
    ori_len = 0
    ap_et_overhead = lap_et_overhead = 0
    ap_list = []
    lap_list = []
    buffer_q = queue.Queue()
    buffer_p = [] #[time, index, size]
    proc_q = queue.Queue() #[buffered_time, buffered_index, size, cleaned_time, cleaned_index, dummy_n, real_n]
    return_q = queue.Queue()
    n = [0, 0, 0, 0]
    ap_list.append(n)
    lap_list.append(n)
    diff_o = 0
    diff_d = 0
    for i, p in enumerate(ori_packets):
        if i == len(ori_packets) - 1:
            continue
        if p[0] < end_time:
            end_time = extension(p[0]) + 4.0
            end_time = round(end_time, 3)

            ap_list.append(p)
            (lap_p, diff_o) = lap(ap_list, lap_list, eps)
            if lap_p[1] == 0:
                buffer_q.put([lap_p[0], len(lap_list) - 1, abs(diff_o)])
                continue
            else:
                lap_list.append(lap_p)

                dummy_n = real_n = 0
                a=buffer_q.qsize()
                b=proc_q.qsize()
                size_left = lap_p[1]
                size_use = 0
                if not buffer_q.empty() or buffer_p:

                    while size_left > 0:
                        if buffer_p:
                            pass
                        else:
                            try:
                                buffer_p = buffer_q.get(False)
                            except queue.Empty:
                                break

                        if size_left > buffer_p[2]:
                            real_n += 1
                            proc_q.put(buffer_p + [lap_p[0], len(lap_list) - 2] + [dummy_n, real_n])
                            size_use += buffer_p[2]
                            size_left = size_left - buffer_p[2]
                            buffer_p = []
                            real_n = dummy_n = 0
                            try:
                                buffer_p = buffer_q.get(False)
                            except queue.Empty:
                                # buffer_p = []
                                if diff_o > 0:
                                    if size_left >= diff_o:
                                        buffer_q.put([lap_p[0], len(lap_list) - 2, size_use])
                                        lap_overhead = lap_overhead + diff_o
                                    else:
                                        buffer_q.put([lap_p[0], len(lap_list) - 2, lap_p[1] - diff_o])
                                        lap_overhead = lap_overhead + diff_o - size_left
                                else:
                                    buffer_q.put([lap_p[0], len(lap_list) - 2, size_use + abs(diff_o)])
                                size_left = 0
                                break
                        else:
                            buffer_p[2] = buffer_p[2] - size_left
                            if size_left > diff_o > 0:
                                buffer_q.put([lap_p[0], len(lap_list) - 2, size_left - diff_o])
                            elif diff_o < 0:
                                buffer_q.put([lap_p[0], len(lap_list) - 2, size_left + abs(diff_o)])
                            real_n += 1
                            size_left = 0

                elif diff_o < 0:
                    buffer_q.put([lap_p[0], len(lap_list) - 2, abs(diff_o)])


                (size, interval) = sample_from_distribution(size_list, interval_list)
                gap = abs(p[0] - ori_packets[i + 1][0])

                while gap > interval and p[0] + interval <= ori_packets[i + 1][0] and size > 0:
                    dummy = [ap_list[-1][0] + interval, size, p[2], 'dummy']

                    ap_list.append(dummy)

                    (lap_p, diff_d) = lap(ap_list, lap_list, eps)
                    if lap_p[1] == 0:
                        pass
                    else:
                        lap_p.append('dummy')
                        lap_list.append(lap_p)

                        if not buffer_q.empty() or buffer_p:
                            if buffer_p :
                                pass
                            else:
                                buffer_p = buffer_q.get()
                            size_d = lap_p[1]
                            while size_d > buffer_p[2]:
                                dummy_n += 1
                                proc_q.put(buffer_p + [lap_p[0], len(lap_list)-2] + [dummy_n, real_n])
                                size_d = size_d - buffer_p[2]
                                dummy_n = real_n = 0
                                try:
                                    buffer_p = buffer_q.get(False)
                                except queue.Empty:
                                    buffer_p = []
                                    lap_overhead = lap_overhead + size_d
                                    break
                            if buffer_p:
                                buffer_p[2] = buffer_p[2] - abs(size_d)
                                dummy_n += 1
                        else:
                            lap_overhead += lap_p[1]

                    ap_overhead = ap_overhead + size
                    gap = abs(ap_list[-1][0] - ori_packets[i + 1][0])
                    (size, interval) = sample_from_distribution(size_list, interval_list)
        else:
            ori_end = ori_packets[-1][0]
            print(p[0] - ori_packets[-1][0])
            unfinished = True
            break
    if ap_list[-1][0] < end_time:
        (ap_et_overhead, lap_et_overhead, return_q) = extend_trace_lap(ap_list,lap_list, size_list, interval_list, end_time, buffer_q, buffer_p, proc_q, eps)
    else:
        return_q = proc_q
    ori_len = len(ori_packets)
    return ap_list, lap_list, ap_overhead, lap_overhead, ap_et_overhead, lap_et_overhead, unfinished, return_q, ori_len


def info_stat(eps, trace_name, ori_size, real_ap_overhead, et_ap_overhead, ap_overall_overhead, real_lap_overhead, et_lap_overhead, lap_overall_overhead, ori_end, overall_delay, unfinished):

    with open('stats/overhead_list_' + str(eps) + '_in.csv','a') as build:
        writer = csv.writer(build)
        writer.writerow([trace_name, ori_size, real_ap_overhead, et_ap_overhead, ap_overall_overhead,
                         real_lap_overhead, et_lap_overhead, lap_overall_overhead, ori_end, overall_delay, unfinished])
    


def distribution_generator(packets, trace_name, folder, eps):
    
    in_size_path = 'stats/distribution_gamma/adapt_in_distribution_size.csv'
    in_interval_path = 'stats/distribution_gamma/adapt_in_distribution_interval.csv'
    in_size_list = su.csv_numpy(in_size_path)
    in_interval_list = su.csv_numpy(in_interval_path)

    in_size_list.sort(key=su.sort_by_second, reverse=True)
    in_interval_list.sort(key=su.sort_by_second, reverse=True)
    in_size_list = in_size_list[0:100]
    calculate_ratio(in_size_list)
    calculate_ratio(in_interval_list)

    ori_end = packets[-1][0]

    ori_size = 0
   

    outgoing = []
    incoming = []
    for p in packets:
        if p[2] == -1:
            incoming.append(p)
        else:
            outgoing.append([p[0], 0, p[2]])
    for p in incoming:
        ori_size = ori_size + p[1]
    (outgoing_ap, outgoing_lap, out_ap_overhead, out_lap_overhead, out_ap_et_overhead,
    out_lap_et_overhead, out_unfinished, out_proc_q, positive_1) = fill_gap_lap(outgoing, out_size_list, out_interval_list, eps)

    (incoming_ap, incoming_lap, in_ap_overhead, in_lap_overhead, in_ap_et_overhead,
     in_lap_et_overhead, in_unfinished, in_proc_q, positive_2) = fill_gap_lap(incoming, in_size_list, in_interval_list, eps)


    outgoing_lap.pop(0)
    incoming_lap.pop(0)
    outgoing_ap.pop(0)
    incoming_ap.pop(0)
    buffer_list = list(out_proc_q.queue) + list(in_proc_q.queue)
    buffer_list.sort(key=su.sort_by_name)
    # # buffer_list.append([positive_1 + positive_2])
    buffer_list.append([len(incoming_lap) + len(outgoing_lap)])
    buffer_list = list(in_proc_q.queue)
    try:
        buffer_df = pd.DataFrame(buffer_list, columns = ['buffered_time', 'buffered_index', 'size', 'cleaned_time', 'cleaned_index', 'dummy_n', 'real_n'])
        buffer_df.to_csv('/home/lhp/PycharmProjects/2019_spring_data/optionB/' + str(eps) + '/' + folder + '/' + trace_name + 'buffer.csv', index=False)
    except AssertionError:
        print('no proc queue!!!')

    ap_list = outgoing_ap + incoming_ap
    ap_list.sort(key=su.sort_by_name)
    real_ap_overhead = out_ap_overhead + in_ap_overhead
    real_lap_overhead = out_lap_overhead + in_lap_overhead
    et_ap_overhead = out_ap_et_overhead + in_ap_et_overhead
    et_lap_overhead = out_lap_et_overhead + in_lap_et_overhead
    ap_overall_overhead = real_ap_overhead + et_ap_overhead
    lap_overall_overhead = real_lap_overhead + et_lap_overhead
    unfinished = out_unfinished or in_unfinished

    real_ap_overhead =  in_ap_overhead
    real_lap_overhead =  in_lap_overhead
    et_ap_overhead =  in_ap_et_overhead
    et_lap_overhead =  in_lap_et_overhead
    ap_overall_overhead = real_ap_overhead + et_ap_overhead
    lap_overall_overhead = real_lap_overhead + et_lap_overhead
    unfinished =  in_unfinished

    # # # ap_df = pd.DataFrame(ap_list, columns=['time', 'size', 'direction', 'type'])
    # # # ap_df.to_csv('obf_data/adapt_list/'+ folder + '/' + trace_name + '_ap.csv', index=False)
    # #
    lap_list = incoming_lap + outgoing
    lap_list.sort(key=su.sort_by_name)
    info_stat(eps, trace_name, ori_size, real_ap_overhead, et_ap_overhead, ap_overall_overhead,
              real_lap_overhead, et_lap_overhead, lap_overall_overhead, ori_end, lap_list[-1][0] - ori_end, unfinished)
    lap_df = pd.DataFrame(lap_list, columns=['time', 'size', 'direction', 'type'])

    if not os.path.isdir('obf_data/lap_list/' + str(eps) + '/' + folder):
        os.makedirs('obf_data/lap_list/' + str(eps) + '/' + folder)
    lap_df.to_csv('obf_data/lap_list/' + str(eps) + '/' + folder + '/' + trace_name + 'lap.csv', index=False)


def main(opt):
  
    csv_path = opts.csvPath
    folder = opts.folder

    eps = float(opts.eps)

    dst = '/home/lhp/PycharmProjects/2019_spring_data/optionB/' + str(eps) + '/' + folder

    if not os.path.isdir(dst):
        os.makedirs(dst)

    packets = su.csv_numpy(csv_path)
    pf = Path(csv_path)
    trace_name = pf.name[0:-4]
    distribution_generator(packets, trace_name, folder, eps)
    print(csv_path + ' is finished')


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-pc','--csvPath', help='path to read csv files')
    parser.add_argument('-f','--folder', help='obf data folder')
    parser.add_argument('-eps','--eps', help='eps for laplase' )
    opts = parser.parse_args()
    return opts

if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
