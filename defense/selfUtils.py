import csv
import pandas as pd
import re
import traceback



def sort_by_second(elem):
    return int(elem[1])


def sort_by_third(elem):
    return int(elem[2])


def sort_by_fourth(elem):
    return int(elem[3])


def sort_by_fifth(elem):
    return int(elem[4])


def sort_by_sixth(elem):
    return int(elem[5])


def sort_by_name(elem):
    return elem[0]


def sort_by_last(elem):
    return elem[-1]


def csv_numpy(packet_path):
    """
    Take the csv files, convert it to "list" structure or "ndarray" structure. Use either of the two
    structure for the future data analysis.
    :param packet_path: path of the csv file
    :return: return the traffic data in a data structure of "list" or "ndarray"
    """
    # df = pd.read_csv(packet_path)
    reader = csv.reader(open(packet_path, "r"), delimiter=",")
    query_in_list = list(reader)
    # query_in_list = df.values.tolist()

    try:
        float(query_in_list[0][1])
    except ValueError:
        query_in_list.pop(0)
    except IndexError:
        print("empty csv")

    N = []
    for p in query_in_list:
        new_list = []

        for m in p:
            try:
                new_list = new_list + [float(m)]
            except ValueError:
                new_list = new_list + [m]
                pass
        N.append(new_list)

    return N


def cal_d(i):
    for j in range(10000):
        if i%pow(2, j) == 0 and i%pow(2, j+1) != 0:
            d = pow(2, j)
            return int(d)


def cal_g(i):
    g = 0
    if i == 1:
        g = 0
    else:
        d = cal_d(i)
        if i == d and d >= 2:
            g = i/2
        elif i > d:
            g = i - d
    return int(g)


def same_name(string1, string2):
    name1 = 0
    name2 = 0

    for a, x in enumerate(string1):
        # if x.isdigit():
        if x == '?' and string1[a-1] == '_':
            name1 = string1[0:a-1]
            break
        else:
            continue
    for b, y in enumerate(string2):
        # if x.isdigit():
        if y == '?' and string2[b-1] == '_':
            name2 = string2[0:b-1]
            break
        else:
            continue

    return name1 == name2


def extract_name(string):
    for a, x in enumerate(string):
        # if x.isdigit():
        if x == '?' and string[a-1] == '_':
            name = string[0:a-1]
            break
        else:
            continue
    return name


def just_class(file_path):
    data = pd.read_csv(file_path, index_col=0)
    labels_raw = data['class']
    labels = []
    for l in labels_raw:
        try:
            match = re.search("\\?", l)
            class_name = l[:match.start()]
            labels.append(class_name)
        except AttributeError:
            print(l)
            print(traceback.print_exc())
    data['class'] = labels
    data.to_csv('stats/Gamma_class.csv')



# def fill_gap_lap_o(ori_packets, size_list, interval_list, eps):
#     unfinished = False
#     end_time = 100.0
#     ap_overhead = 0.0
#     lap_overhead = 0.0
#     positive_n = 0
#     # buff = 0
#     ap_et_overhead = lap_et_overhead = 0
#     ap_list = []
#     lap_list = []
#     buffer_q = queue.Queue()
#     buffer_p = [] #[time, index, size]
#     proc_q = queue.Queue() #[buffered_time, buffered_index, size, cleaned_time, cleaned_index, dummy_n, real_n]
#     return_q = queue.Queue()
#     n = [0, 0, 0, 0]
#     ap_list.append(n)
#     lap_list.append(n)
#     diff_o = 0
#     diff_d = 0
#     for i, p in enumerate(ori_packets):
#         if i == len(ori_packets) - 1:
#             continue
#         if p[0] < end_time:
#             end_time = extension(p[0]) + 4.0
#             end_time = round(end_time, 3)
#
#             ap_list.append(p)
#             (lap_p, diff_o) = lap(ap_list, lap_list, eps)
#
#             lap_list.append(lap_p)
#             dummy_n = real_n = 0
#             if diff_o < 0:
#                 buffer_q.put([lap_p[0], len(ap_list)-2, abs(diff_o)])
#             elif diff_o > 0:
#                 positive_n += 1
#                 if not buffer_q.empty() or buffer_p:
#                     # buffer_p = lambda b: b if b else (b = buffer_q.get())
#                     if buffer_p:
#                         pass
#                     else:
#                         buffer_p = buffer_q.get()
#
#                     while diff_o >= buffer_p[2]:
#                         real_n += 1
#                         # buffer_p[3] = i
#                         proc_q.put(buffer_p + [lap_p[0], len(lap_list)-2] + [dummy_n, real_n])
#                         diff_o = max(diff_o - buffer_p[2], 0)
#                         buffer_p = []
#                         real_n = dummy_n = 0
#                         try:
#                             buffer_p = buffer_q.get(False)
#                         except queue.Empty:
#
#                             lap_overhead = lap_overhead + diff_o
#                             break
#                     if buffer_p:
#                         buffer_p[2] = buffer_p[2] - abs(diff_o)
#                         # buffer_p[3] = i
#                         real_n += 1
#                 else:
#                     lap_overhead = lap_overhead + diff_o
#             elif diff_o == 0:
#                 positive_n += 1
#
#             (size, interval) = sample_from_distribution(size_list, interval_list)
#             gap = abs(p[0] - ori_packets[i + 1][0])
#
#             while gap > interval and p[0] + interval <= ori_packets[i + 1][0]:
#                 dummy = [ap_list[-1][0] + interval, size, p[2], 'dummy']
#                 ap_list.append(dummy)
#
#                 (lap_p, diff_d) = lap(ap_list, lap_list, eps)
#
#                 lap_p.append('dummy')
#                 lap_list.append(lap_p)
#
#                 if not buffer_q.empty() or buffer_p:
#                     if buffer_p :
#                         pass
#                     else:
#                         buffer_p = buffer_q.get()
#                     size_d = lap_p[1]
#                     while size_d > buffer_p[2]:
#                         dummy_n += 1
#                         proc_q.put(buffer_p + [lap_p[0], len(lap_list)-2] + [dummy_n, real_n])
#                         size_d = size_d - buffer_p[2]
#                         dummy_n = real_n = 0
#                         try:
#                             buffer_p = buffer_q.get(False)
#                         except queue.Empty:
#                             buffer_p = []
#                             lap_overhead = lap_overhead + size_d
#                             break
#                     if buffer_p:
#                         buffer_p[2] = buffer_p[2] - abs(size_d)
#                         dummy_n += 1
#                 else:
#                     lap_overhead += lap_p[1]
#
#                 ap_overhead = ap_overhead + size
#                 gap = abs(ap_list[-1][0] - ori_packets[i + 1][0])
#                 (size, interval) = sample_from_distribution(size_list, interval_list)
#         else:
#             ori_end = ori_packets[-1][0]
#             print(p[0] - ori_packets[-1][0])
#             unfinished = True
#             break
#     if ap_list[-1][0] < end_time:
#         # et_overhead = extend_trace(ap_list, size_list, interval_list, end_time)
#         (ap_et_overhead, lap_et_overhead, return_q) = extend_trace_lap(ap_list,lap_list, size_list, interval_list, end_time, buffer_q, buffer_p, proc_q, eps)
#         # return_q.put([positive_n])
#     # delay = ap_list[-1][0] - ori_packets[-1][0]
#     return ap_list, lap_list, ap_overhead, lap_overhead, ap_et_overhead, lap_et_overhead, unfinished, return_q, positive_n