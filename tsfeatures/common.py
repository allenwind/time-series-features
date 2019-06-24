import numpy as np
import scipy.stats as stats
import tsfresh.feature_extraction.feature_calculators as tsfc

def time_series_maximum(series):
    return np.max(series)

def time_series_minimum(series):
    return np.min(series)

def time_series_range(series):
    return np.max(series) - np.min(series)

def time_series_mean(series):
    return np.mean(series)

def time_series_median(series):
    return np.median(series)

def time_series_mode(series):
    return stats.mode(series)

def time_series_standard_deviation(series):
    return np.std(series)

def time_series_variance(series):
    return np.var(series)

def time_series_kurtosis(series):
    return stats.kurtosis(series)

def time_series_skewness(series):
    return stats.skew(series)

def time_series_length(series):
    return len(series)

def time_series_gmean(series):
    return stats.gmean(series)

def time_series_hmean(series):
    # 预处理数据时不能以零为中心
    return stats.hmean(series)

def time_series_coefficient_of_variation(series):
    # scipy.stats.variation
    # wiki:
    # https://en.wikipedia.org/wiki/Coefficient_of_variation
    s = np.sqrt(np.var(series))
    if s < 1e-10:
        return 0
    return np.mean(series) / s

def time_series_cumsum(series):
    # 时序的累加
    return np.cumsum(series)

def time_series_abs_cumsum(series):
    return np.cumsum(np.abs(series))

def time_series_abs_energy(series):
    # 信号处理中的一个概念
    # wiki:
    # https://en.wikipedia.org/wiki/Energy_(signal_processing)
    return np.sum(np.square(series))

def time_series_mean_spectral_energy(series):
    # 时间序列的谱能量
    return np.mean(np.square(np.abs(np.fft.fft(series))))

def time_series_count_above_mean(x):
    return tsfc.count_above_mean(x)

def time_series_count_below_mean(x):
    return tsfc.count_below_mean(x)

def time_series_mean_change(series):
    # 度量时序的变化
    return np.mean(np.diff(series))

def time_series_absolute_mean_of_changes(series):
    # 度量时序的变化
    return np.mean(np.abs(np.diff(series)))

def time_series_mean_second_derivative_central(series):
    # 度量时序的变化
    # wiki:
    # https://en.wikipedia.org/wiki/Finite_difference#Higher-order_differences
    return tsfc.mean_second_derivative_central(series)

class time_series_willison_amplitude:
    
    # 度量时序的变化程度
    # 当 threshold = 0
    # 等价于 time_series_absolute_sum_of_changes

    def __init__(self, threshold=0):
        self.threshold = threshold

    def __call__(self, series):
        return np.sum(np.abs(np.diff(series)) >= self.threshold)

def time_series_absolute_sum_of_changes(series):
    # 度量时序的变化
    # alias time_series_waveform_length
    # np.sum(np.abs(np.diff(series))) equal to 
    # time_series_willison_amplitude(threshold=0)(series)

    return time_series_willison_amplitude(threshold=0)(series)

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

class time_series_zero_crossing:
    
    def __init__(self, threshold=0):
        self.threshold = threshold

    def __call__(self, series):
        sign = np.heaviside(series[1:], 0)
        return np.sum(sign * np.abs(np.diff(series)) >= self.threshold)

class time_series_autocorrelation:

    # 时序的自相关系数

    def __init__(self, lag):
        self.lag = lag

    def __call__(self, series):
        return tsfc.autocorrelation(series, self.lag)

class time_series_agg_autocorrelation:

    def __init__(self, maxlag, func):
        if func not in [np.mean, np.median, np.var, np.max]:
            raise ValueError(func)
        self.maxlag = maxlag
        self.func = func

    def __call__(self, series):
        values = []
        for lag in range(1, self.maxlag):
            value = tsfc.autocorrelation(series, lag)
            values.append(value)
        return self.func(np.array(values))

class time_series_partial_autocorrelation:

    # 时序的偏差自相关系数
    # wiki
    # https://en.wikipedia.org/wiki/Partial_autocorrelation_function

    def __init__(self, lag):
        self.lag = lag

    def __call__(self, series):
        return tsfc.partial_autocorrelation(series, [{"lag": self.lag}])

def time_series_binned_autocorrelation(series):
    max_bins = [2, 3, 4, 5, 6, 7]
    size = len(series) - 3
    values = []
    for value in max_bins:
        lag = size // value
        r = time_series_autocorrelation(lag)(series)
        if r is np.nan:
            r = 0
        values.append(r)
    return values

def time_series_periodic_features(series):
    # TODO
    return []

def time_series_binned_entropy(series):
    max_bins = [2, 4, 6, 8, 10, 20]
    values = []
    for value in max_bins:
        values.append(tsfc.binned_entropy(series, value))
    return values

def time_series_value_distribution(series):
    # 数值需要变换到 0, 1 区间上
    thresholds = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0, 1.0]
    return list(np.histogram(series, bins=thresholds)[0] / float(len(series)))

def extract_time_series_features(series):
    features = []

    # 基本统计特征
    features.append(time_series_minimum(series))
    features.append(time_series_maximum(series))
    features.append(time_series_range(series))
    features.append(time_series_mean(series))
    features.append(time_series_median(series))
    features.append(time_series_mode(series))
    features.append(time_series_standard_deviation(series))
    features.append(time_series_variance(series))
    features.append(time_series_kurtosis(series))
    features.append(time_series_skewness(series))
    features.append(time_series_length(series))
    features.append(time_series_gmean(series))
    # features.append(time_series_hmean(series))
    features.append(time_series_coefficient_of_variation(series))

    # 中心特征 or location
    features.append(time_series_cumsum(series))
    features.append(time_series_abs_cumsum(series))
    features.append(time_series_abs_energy(series))
    features.append(time_series_count_above_mean(series))
    features.append(time_series_count_below_mean(series))

    # 度量变化或分散
    features.append(time_series_mean_change(series))
    features.append(time_series_absolute_mean_of_changes(series))
    features.append(time_series_absolute_sum_of_changes(series))
    features.append(time_series_mean_second_derivative_central(series))
    features.extend(time_series_mean_absolute_deviation_around_a_central_point("all")(series))
    features.extend(time_series_median_absolute_deviation_around_a_central_point("all")(series))
    features.append(time_series_zero_crossing(threshold=0)(series))

    # 自相关和周期
    features.extend(time_series_binned_autocorrelation(series))
    features.extend(time_series_periodic_features(series))

    # 其他
    features.extend(time_series_binned_entropy(series))
    features.extend(time_series_value_distribution(series))

    features = np.array(features)
    return features

def compute_features_size(window_size=100):
    series = np.random.uniform(0.1, 10, window_size)
    return len(extract_time_series_features(series))

FEATURE_SIZE = compute_features_size() # 57
