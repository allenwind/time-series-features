import numpy as np
import scipy.stats as stats
import tsfresh.feature_extraction.feature_calculators as tsfc


# 自相关和时序周期有关的特征


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
        values.append(r)
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

    # 时序的偏差自相关系数
    # wiki
    # https://en.wikipedia.org/wiki/Partial_autocorrelation_function

    def __init__(self, lag):
        self.lag = lag

    def __call__(self, series):
        return tsfc.partial_autocorrelation(series, [{"lag": self.lag}])


def time_series_periodic_features(series):
    # TODO
    return []
