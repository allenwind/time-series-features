import itertools
import collections

import numpy as np
import scipy.signal as signal

# 和 peak 有关的特征, 通常用在时间序列分类和异常检测中

# 添加 I/O latency 相关的峰值特征, 参考 paper:
# Finding soon-to-fail disks in a haystack 
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.564.7041&rep=rep1&type=pdf

def _successive_groupby(bools):
    # 输入布尔序列
    # 然后进行连续分段
    # 比如 bools = [0, 1, 1, 0, 0]
    # 那么结果为 [1, 2, 2]
    cs = [len(list(g)) for v, g in itertools.groupby(x) if v]
    cs.append(0) # 避免输出空 array
    return np.array(cs)

def time_series_count_above_mean(series):
    m = np.mean(series)
    return np.where(series > m)[0].size

def time_series_count_below_mean(series):
    m = np.mean(series)
    return np.where(series < m)[0].size

def time_series_longest_strike_above_mean(series):
    return np.max(_successive_groupby(series >= np.mean(series))) if series.size > 0 else 0

def time_series_longest_strike_below_mean(series):
    return np.max(_successive_groupby(series <= np.mean(series))) if series.size > 0 else 0

class time_series_number_peaks:

    def __init__(self, n):
        self.n = n

    def __call__(self, series):
        n = self.n
        x_reduced = series[n:-n]

        res = None
        for i in range(1, n + 1):
            result_first = (x_reduced > np.roll(series, i)[n:-n])

            if res is None:
                res = result_first
            else:
                res &= result_first

            res &= (x_reduced > np.roll(series, -i)[n:-n])
        return np.sum(res)

class time_series_number_cwt_peaks:

    def __init__(self, n):
        self.n = n

    def __call__(self, series):
        return len(signal.find_peaks_cwt(vector=series, widths=np.array(list(range(1, self.n + 1))), wavelet=signal.ricker))

class time_series_number_peaks_over_k_standard_deviations:

    # peak 持续计数
    # 连续 d 个时间步取值高于 k 个标准差的计数
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
                if n >= d:
                    count += 1
            else:
                n = 0
        return count

def extract_time_series_peak_features(series):
    features = []
    features.append(time_series_number_peaks(2)(series))
    features.append(time_series_number_cwt_peaks(2)(series))
    return features
