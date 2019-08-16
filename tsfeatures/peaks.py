import itertools
import collections

import numpy as np
import scipy.signal as signal

# 和 peak 有关的特征, 通常用在时间序列分类和异常检测中

# TODO 添加 I/O latency 相关的峰值特征, 参考 paper:
# Finding soon-to-fail disks in a haystack 
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.564.7041&rep=rep1&type=pdf

def _get_length_sequences_where(x):
    res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
    return res if len(res) > 0 else [0]

def time_series_count_above_mean(x):
    m = np.mean(x)
    return np.where(x > m)[0].size

def time_series_count_below_mean(x):
    m = np.mean(x)
    return np.where(x < m)[0].size

def time_series_longest_strike_above_mean(series):
    return np.max(_get_length_sequences_where(x >= np.mean(x))) if x.size > 0 else 0

def time_series_longest_strike_below_mean(series):
    return np.max(_get_length_sequences_where(x <= np.mean(x))) if x.size > 0 else 0

def time_series_number_peaks(series, n):
    x_reduced = x[n:-n]

    res = None
    for i in range(1, n + 1):
        result_first = (x_reduced > np.roll(x, i)[n:-n])

        if res is None:
            res = result_first
        else:
            res &= result_first

        res &= (x_reduced > np.roll(x, -i)[n:-n])
    return np.sum(res)

def time_series_number_cwt_peaks(series, n):
    return len(signal.find_peaks_cwt(vector=x, widths=np.array(list(range(1, n + 1))), wavelet=signal.ricker))

class time_series_number_peaks_over_k_standard_deviations:

    # peak 持续计数
    # 持续 d 个时间步取值高于 k 个标准差的计数
    # 磁盘的 I/O Latency 统计需要这个特征

    def __init__(self, k, d):
        self.k = k
        self.d = d

    def __call__(self, series):
        k_std = k * np.std(series)
        count = 0

        n = 0
        for v in series:
            if v >= k_std:
                n += 1
                if n > d:
                    count += 1
            else:
                n = 0
        return count

def extract_time_series_peak_features(series):
    features = []

    features.append(time_series_number_peaks(series,2))
    features.append(time_series_number_cwt_peaks(series, 2))
    return features
