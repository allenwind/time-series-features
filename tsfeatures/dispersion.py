import numpy as np
import scipy.stats as stats
import tsfresh.feature_extraction.feature_calculators as tsfc


# 度量时序的分散情况
# 可以参考如下资料
# https://en.wikipedia.org/wiki/Statistical_dispersion


def time_series_range(series):
    return np.max(series) - np.min(series)

def time_series_count_above_mean(x):
    return tsfc.count_above_mean(x)

def time_series_count_below_mean(x):
    return tsfc.count_below_mean(x)

def time_series_longest_strike_above_mean(series):
    return tsfc.longest_strike_above_mean(series)

def time_series_longest_strike_below_mean(series):
    return tsfc.longest_strike_below_mean(series)

class time_series_over_k_sigma_count:
    
    def __init__(self, k):
        self.k = k

    def __init__(self, series):
        c = 0
        k_std = k * np.std(series)
        mean = np.mean(series)

        for v in series:
            if not ((-k_std+mean) <= v <= (k_std+mean)):
                c += 1
        return c

def time_series_ratio_beyond_r_sigma(series, r):
    return tsfc.ratio_beyond_r_sigma(series, r)

def time_series_median_absolute_deviation(series):
    # 度量时序的变化
    # median absolute deviation (MAD)
    # 当使用 time_series_median_absolute_deviation_around_a_central_point("all")
    # 不要使用这特征
    # wiki:
    # https://en.wikipedia.org/wiki/Median_absolute_deviation
    # https://en.wikipedia.org/wiki/Statistical_dispersion

    return np.median(np.abs(series-np.median(series)))

class time_series_mean_absolute_deviation_around_a_central_point:
    # 度量时序的变化
    # 以m(x)为中心度量时序的变化(分散)情况, 包括 mean(均值) median(中位数) mode(众数)
    # 这类特征可以约束 forecast 中的波动. 在 changepoints 检测中也有运用.
    # 这里的 central point 包括:
    # mean median mode max

    # 相关文档可参考
    # wiki:
    # https://en.wikipedia.org/wiki/Average_absolute_deviation
    # Statistical dispersion
    # https://en.wikipedia.org/wiki/Statistical_dispersion

    def __init__(self, m):
        if m not in ["mean", "median", "mode", "max", "all"]:
            raise KeyError()

        if m == "mean":
            self.ms = [np.mean]
        elif m == "median":
            self.ms = [np.median]
        elif m == "mode":
            self.ms = [stats.mode]
        elif m == "max":
            self.ms = [np.max]
        else:
            self.ms = [np.mean, np.median, stats.mode]

    def __call__(self, series):
        values = []
        for m in self.ms:
            value = np.mean(np.abs(series-m(series)))
            values.append(value)
        return values

class time_series_median_absolute_deviation_around_a_central_point(
    time_series_mean_absolute_deviation_around_a_central_point):

    # 继承自 time_series_mean_absolute_deviation_around_a_central_point
    # 但是用 median 替代 mean

    def __call__(self, series):
        values = []
        for m in self.ms:
            value = np.median(np.abs(series-m(series)))
            values.append(value)
        return values
