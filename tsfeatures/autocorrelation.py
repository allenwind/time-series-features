import numpy as np
import scipy.stats as stats
from statsmodels.tsa.stattools import pacf

# 自相关和时序周期有关的特征
# TODO fft

class time_series_autocorrelation:
    
    # 时序的自相关系数

    def __init__(self, lag):
        self.lag = lag

    def __call__(self, series):
        v = np.var(series)

        # 如果方差太接近 0 自相关系数没有意义
        if np.isclose(v, 0):
            return np.nan

        y1 = series[:series.size-self.lag]
        y2 = series[self.lag:]
        m = np.mean(series)
        return np.sum((y1 - m) * (y2 - m)) / ((series.size - self.lag) * v)

def time_series_all_autocorrelation(series):
    # 计算所有 lag 的自相关值, 计算方法可参考
    # wiki:
    # https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation

    n = series.size
    rs = []
    for i in range(n):
        r = 0
        for j in range(n-i):
            r += series[j+i] * series[j]
        rs.append(r)
    return rs

class time_series_partial_autocorrelation:

    # 时序的偏自相关系数
    # wiki:
    # https://en.wikipedia.org/wiki/Partial_autocorrelation_function

    def __init__(self, lag):
        self.lag = lag

    def __call__(self, series):
        max_demanded_lag = max([lag["lag"] for lag in param])
        n = len(x)

        # Check if list is too short to make calculations
        if n <= 1:
            pacf_coeffs = [np.nan] * (max_demanded_lag + 1)
        else:
            if (n <= max_demanded_lag):
                max_lag = n - 1
            else:
                max_lag = max_demanded_lag
            pacf_coeffs = list(pacf(x, method="ld", nlags=max_lag))
            pacf_coeffs = pacf_coeffs + [np.nan] * max(0, (max_demanded_lag - max_lag))

        return [("lag_{}".format(lag["lag"]), pacf_coeffs[lag["lag"]]) for lag in param]


def time_series_binned_autocorrelation(series):
    max_bins = [2, 3, 4, 5, 6, 7]
    size = len(series) - 3
    values = []
    for value in max_bins:
        lag = size // value
        c = time_series_autocorrelation(lag)(series)
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
            value = time_series_autocorrelation(series, lag)
            values.append(value)
        return self.func(np.array(values))

def time_series_binned_partial_autocorrelation(series):
    max_bins = [2, 3, 4, 5, 6, 7]
    size = len(series) - 3
    values = []
    for value in max_bins:
        lag = size // value
        c = time_series_partial_autocorrelation(series, [{"lag": lag}])[0][1]
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
