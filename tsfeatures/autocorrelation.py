import numpy as np
import scipy.stats as stats
import tsfresh.feature_extraction.feature_calculators as tsfc

# 自相关和时序周期有关的特征

def time_series_all_autocorrelation(series):
    # 计算所有 lag 的自相关值, 计算方法可参考
    # wiki
    # https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation

    N = len(series)
    auto = np.zeros(N)
    for k in range(N):
        for n in range(N - k):
            auto[k] += series[n + k] * series[n]
    return auto.tolist()

class time_series_autocorrelation:
    
    # 时序的自相关系数

    def __init__(self, lag):
        self.lag = lag

    def __call__(self, series):
        return tsfc.autocorrelation(series, self.lag)

def time_series_binned_autocorrelation(series):
    max_bins = [2, 3, 4, 5, 6, 7]
    size = len(series) - 3
    values = []
    for value in max_bins:
        lag = size // value
        c = tsfc.autocorrelation(series, lag)
        if c is np.nan:
            c = 0
        values.append(c)
    return values

class time_series_agg_autocorrelation:

    def __init__(self, maxlag, func=None):
        if func is None:
            func = np.mean

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

    # 时序的偏自相关系数
    # wiki
    # https://en.wikipedia.org/wiki/Partial_autocorrelation_function

    def __init__(self, lag):
        self.lag = lag

    def __call__(self, series):
        return tsfc.partial_autocorrelation(series, [{"lag": self.lag}])

def time_series_binned_partial_autocorrelation(series):
    max_bins = [2, 3, 4, 5, 6, 7]
    size = len(series) - 3
    values = []
    for value in max_bins:
        lag = size // value
        c = tsfc.partial_autocorrelation(series, [{"lag": lag}])[0][1]
        if c is np.nan:
            c = 0
        values.append(c)
    return values

def time_series_periodic_features(series):
    return  []

def extract_time_series_autocorrelation_based_features(series):
    features = []

    features.extend(time_series_all_autocorrelation(series))
    features.extend(time_series_binned_partial_autocorrelation(series))
    return features
