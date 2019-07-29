import numpy as np
import scipy.stats as stats

# 度量时序的分散情况
# 可以参考如下资料
# https://en.wikipedia.org/wiki/Statistical_dispersion

def time_series_location_range(series):
    return series[-1] - series[0]

def time_series_range(series):
    return np.max(series) - np.min(series)

def time_series_center(series):
    # 极值中心
    return (np.max(series) + np.min(series)) / 2

def time_series_has_duplicate(series):
    # 有冗余
    return series.size != np.unique(series).size

def time_series_has_duplicate_min(series):
    return np.sum(series == np.min(series)) >= 2

def time_series_has_duplicate_max(series):
    return np.sum(series == np.max(series)) >= 2

def time_series_ratio_of_unique_number_to_length(series):
    return np.unique(series).size / series.size

class time_series_large_standard_deviation:

    def __init__(self, r):
        self.r = r

    def __call__(self, series):
        return np.std(series) > (self.r * time_series_range(series))

class time_series_over_k_sigma_ratio:
    
    def __init__(self, k):
        self.k = k

    def __init__(self, series):
        return np.sum(np.abs(series - np.mean(series)) > r * np.std(series)) / series.size

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
    # mean median mode time_series_center

    # 相关文档可参考
    # wiki:
    # https://en.wikipedia.org/wiki/Average_absolute_deviation
    # Statistical dispersion
    # https://en.wikipedia.org/wiki/Statistical_dispersion

    def __init__(self, m):
        if m not in ["mean", "median", "mode", "max", "all"]:
            raise KeyError(m)

        if m == "mean":
            self.ms = [np.mean]
        elif m == "median":
            self.ms = [np.median]
        elif m == "mode":
            self.ms = [stats.mode]
        elif m == "max":
            self.ms = [np.max]
        else:
            self.ms = [np.mean, np.median, stats.mode, time_series_center]

    def __call__(self, series):
        values = []
        for m in self.ms:
            value = np.mean(np.abs(series-m(series)))
            values.append(value)
        return values

class time_series_median_absolute_deviation_around_a_central_point(
    time_series_mean_absolute_deviation_around_a_central_point):

    # 继承自 time_series_mean_absolute_deviation_around_a_central_point
    # 但是用 median 替代 mean, 使用 median 度量 location 变化的论文可以参考:
    # https://www.sciencedirect.com/science/article/pii/S0167947302000786
    # 磁盘异常检测中也使用这个特征

    def __call__(self, series):
        values = []
        for m in self.ms:
            value = np.median(np.abs(series-m(series)))
            values.append(value)
        return values

def extract_time_series_dispersion_features(series):
    features = []

    features.append(time_series_range(series))
    features.append(time_series_center(series))
    features.append(time_series_has_duplicate(series))
    features.append(time_series_has_duplicate_min(series))
    features.append(time_series_has_duplicate_max(series))

    # features.append(time_series_median_absolute_deviation(series)) duplicate
    features.extend(time_series_mean_absolute_deviation_around_a_central_point("all")(series))
    features.extend(time_series_median_absolute_deviation_around_a_central_point("all")(series))
    
    return features
