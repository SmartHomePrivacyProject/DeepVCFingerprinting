from __future__ import division
import csv
import shutil
import numpy as np
import pandas as pd
import selfUtils as su


def divide_list(packets):
    for i,p in enumerate(packets):
        if float(p[3]) + float(packets[i+1][3]) == 0:
            return i


def add_dummy(packets):
    max_num = int(packets[-1][2])
    for p in packets:
        if int(p[2]) < max_num:
            dummy_num = max_num - int(p[2])
            p.append(dummy_num)

    return packets


def logk_process(filter_in_stats):
    filter_in_stats.sort(key= su.sort_by_name)
    same_traces = []
    for i, p in enumerate(filter_in_stats):
        if i == 0:
            same_traces.append(p)
            continue
        if i == len(filter_in_stats) - 1:
            same_traces.append(p)
            same_traces.sort(key=su.sort_by_third)
            with open("stats/logk_analysis.csv",'a') as _in:
                writer = csv.writer(_in)
                writer.writerow(same_traces[-1])
            # traces_df = pd.DataFrame(same_traces)
            # traces_df.to_csv('stats' + filter_in_stats[i - 1][0] + '.csv')
            continue
        if su.same_name(p[0], filter_in_stats[i-1][0]):
            same_traces.append(p)
        else:
            same_traces.sort(key=su.sort_by_third)
            with open("stats/logk_analysis.csv",'a') as _in:
                writer = csv.writer(_in)
                writer.writerow(same_traces[-1])

            same_traces = []
            same_traces.append(p)

    filter_in_stats = su.csv_numpy("stats/logk_analysis.csv")
    filter_in_stats.sort(key=su.sort_by_third)

    l = len(filter_in_stats)
    k = int(np.ceil(l/2))
    start = 0
    logk_list = []

    while (k >= 1):
        if start + k <= len(filter_in_stats):
            print("index({},{})".format(start, start + k - 1))
            if start == len(filter_in_stats) - 2:
                anonymity_list = filter_in_stats[start:(start + k + 1)]
            else:
                anonymity_list = filter_in_stats[start:(start + k)]
            logk = add_dummy(anonymity_list)
            start = start + k
            logk_list = logk_list + logk
          
        else:
            break
        k = int(np.ceil((l - start) / 2))
        print(len(logk_list))
    echo_df2 = pd.DataFrame(logk_list, columns=['name', 'original_in_num', 'duplex_num', 'remainder', 'padded_num'])
    echo_df2.to_csv("stats/logk_distribution.csv", index=False)


def outgoing_process(stats):
    stats.sort(key=su.sort_by_fourth, reverse=True)
    new_out_stats = []
    index = 0
    for p in stats:
        if int(p[3]) > 1000:

            for i in range(1, 21):
                try:
                    shutil.move('csv/open_world/' + str(i) + "/" + p[0] + "csv", "unqualified_csv/")
                    shutil.move('/home/lhp/PycharmProjects/pcap_csv/half_duplex/open_world/' + str(i) + "/" + p[0] + "_HD.csv",
                                "/home/lhp/PycharmProjects/pcap_csv/unqualified_hd/")
                except IOError:
                    continue

            continue
        else:
            index = stats.index(p)
            break
    end_index = 0
    stats.sort(key=su.sort_by_fourth)
    for p in stats:
        if int(p[3]) < 10:

            for i in range(1, 21):
                try:
                    shutil.move('csv/open_world/' + str(i) + "/" + p[0] + "csv", "unqualified_csv/")
                    shutil.move('/home/lhp/PycharmProjects/pcap_csv/half_duplex/open_world/' + str(i) + "/" + p[0] + "_HD.csv",
                                "/home/lhp/PycharmProjects/pcap_csv/unqualified_hd/")
                except IOError:
                    continue

            continue
        else:
            end_index = stats.index(p)
            break

    stats.sort(key=su.sort_by_fourth, reverse=True)
    filter_out_stats = stats[index:len(stats)-end_index]
    
    return filter_out_stats


def incoming_filter(stats):
    ind1, ind2 = 0, 0
    # unqualified_index = []
    stats.sort(key=su.sort_by_name)
    for p in stats:
        if p[-1] > 10000:
            # for i in range(1, 21):
            #     try:
            #         shutil.move('csv/gamma/' + str(i) + "/" + p[0] + "csv", "unqualified_csv/")
            #         # shutil.move('/home/lhp/PycharmProjects/pcap_csv/half_duplex/gamma/' + str(i) + "/"+ p[0] + "_HD.csv", "/home/lhp/PycharmProjects/pcap_csv/unqualified_hd/")
            #     except IOError:
            #         continue
            # shutil.move('/home/lhp/PycharmProjects/pcap_csv/raw_pcap/amazon_echo/' + p[0][0:-6] + '.pcap', "/home/lhp/PycharmProjects/pcap_csv/unqualified_pcap/")
            p.append("del")
            # unqualified_index.append(stats.index(p))

    filter_in_stats = [x for x in stats if len(x) == 13]
    filter_in_stats.sort(key=su.sort_by_sixth)
    # filter_in_stats = stats
    stats_df = pd.DataFrame(filter_in_stats,
                            columns=['name', 'original_out_num', 'original_in_num', 'duplex_out_num', 'out_remainder', 'duplex_in_num'
                                    ,'in_remainder','max_interval', 'min_interva', 'ave_interval', 'original_total_num', 'ave', 'ratio'])
    

    filter_in_stats = stats_df[['name', 'original_in_num', 'duplex_in_num','in_remainder']]
    filter_in_stats = filter_in_stats.values.tolist()
    return filter_in_stats


def incoming_process(stats):
    stats.sort(key=su.sort_by_name)
    total = 0
    start_ind = 0
    end_ind = 0
    ratio = 0
    

    stats.sort(key=su.sort_by_sixth, reverse=True)
    index = 0
    for p in stats:
        if int(p[5]) > 3000:
            for i in range(1, 21):
                try:
                    shutil.move('csv/open_world/'+ str(i) + "/" + p[0] + "csv", "unqualified_csv/")
                    shutil.move('/home/lhp/PycharmProjects/pcap_csv/half_duplex/open_world/' + str(i) + "/" + p[0] + "_HD.csv", "/home/lhp/PycharmProjects/pcap_csv/unqualified_hd/")
                except IOError:
                    continue

            continue
        else:
            index = stats.index(p)
            break

    stats.sort(key=su.sort_by_sixth)
    e_index = 0
    for p in stats:
        if int(p[5]) < 1:
            for i in range(1, 21):
                try:
                    shutil.move('csv/open_world/'+ str(i) + "/" + p[0] + "csv", "unqualified_csv/")
                    shutil.move('/home/lhp/PycharmProjects/pcap_csv/half_duplex/open_world/' + str(i) + "/" + p[0] + "_HD.csv", "/home/lhp/PycharmProjects/pcap_csv/unqualified_hd/")
                except IOError:
                    continue

            continue
        else:
            e_index = stats.index(p)
            break

    stats.sort(key=su.sort_by_sixth, reverse=True)
    pre_filter_incoming = stats[index:len(stats)-e_index]
    pre_filter_incoming.sort(key = su.sort_by_name)
    for i,p in enumerate(pre_filter_incoming[end_ind:len(stats)]):
        if i == 0:
            total = total + p[-1]
            continue
        if i == len(pre_filter_incoming) - 1:
            end_ind = pre_filter_incoming.index(p) + 1
            total = total + p[-1]
            ave = total / (end_ind - start_ind)
            total = 0

            for m in pre_filter_incoming[start_ind:end_ind]:
                ratio = m[-1] / ave
                m.append(ave)
                m.append(ratio)
            continue

        if not su.same_name(p[0], pre_filter_incoming[i-1][0]):
            start_ind = pre_filter_incoming.index(p)

        if su.same_name(p[0], pre_filter_incoming[i+1][0]):
            total = total + p[-1]
        else:

            end_ind = pre_filter_incoming.index(p) + 1
            total = total + p[-1]
            ave = total / (end_ind-start_ind)
            total = 0

            for m in pre_filter_incoming[start_ind:end_ind]:
                ratio = m[-1]/ave
                m.append(ave)
                m.append(ratio)

    echo_df2 = pd.DataFrame(pre_filter_incoming,
                            columns=['name', 'original_out_num', 'original_in_num', 'duplex_out_num', 'our_remainder',  'duplex_in_num',
                                     'in_remainder','max_interval', 'min_interva', 'ave_interval', 'original_total_num', 'ave',
                                     'ratio'])
    echo_df2.to_csv("stats/overall_stats.csv", index=False)
    pre_filter_incoming.sort(key=su.sort_by_last)

    filter_in_stats = incoming_filter(pre_filter_incoming)

    logk_process(filter_in_stats)


def main():
    # incoming_stats_path = "/home/lhp/PycharmProjects/pcap_csv/stats/incoming_number.csv"
    # outgoing_stats_path = "/home/lhp/PycharmProjects/pcap_csv/stats/outgoing_number.csv"
    #
    # in_stats = csv_numpy(incoming_stats_path)
    # out_stats = csv_numpy(outgoing_stats_path)
    # in_stats.sort(key=sort_by_second)

    stats_path = "stats/stats.csv"
    stats = su.csv_numpy(stats_path)
    for p in stats:
        num = int(p[3]) + int(p[5])
        p.append(num)


    stats_out = outgoing_process(stats)

    incoming_process(stats_out)


if __name__ == '__main__':
    main()
