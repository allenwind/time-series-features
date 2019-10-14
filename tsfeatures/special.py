import itertools
import numpy as np

# 一些高级特征, 来自某些 paper 或专用与某个场景的特征
# 实现中已经注明 papers.

def time_series_sum(series):
    # 时序的累加
    return np.sum(series)

def time_series_abs_sum(series):
    # 时序改变的累加
    return np.sum(np.abs(series))

def time_series_cid_ce(series):
    # CID distance
    # paper:
    # http://www.cs.ucr.edu/~eamonn/Complexity-Invariant%20Distance%20Measure.pdf

    d = np.diff(series)
    return np.sqrt(np.dot(d, d))

class time_series_c3:

    def __init__(self, lag):
        self.lag = lag

    def __call__(self, series):
        # a measurement of non linearity
        # paper:
        # https://www.macalester.edu/~kaplan/knoxville/PRE05443.pdf

        n = len(series)
        size = n - 2*self.lag
        if size <= 0:
            return 0

        lag1 = np.roll(series, -self.lag)
        lag2 = np.roll(series, -2*self.lag)
        return np.mean((series*lag1*lag2)[0:size])

class time_series_time_reversal_asymmetry_statistic:

    def __init__(self, lag):
        self.lag = lag

    def ___call__(self, series):
        # paper: 
        # Highly comparative feature-based time-series classification
        
        n = len(series)
        size = n - 2*self.lag
        if size <= 0:
            return 0
        lag1 = np.roll(series, -self.lag)
        lag2 = np.roll(series, -2*self.lag)
        return np.mean((lag2*lag2*lag1 - lag1*series*series)[0:size])

class time_series_exponential_compress_representation:

    # 变长序列的压缩表示
    # 可以看作是 RNN 的特征提取过程

    def __init__(self, alpha=0.95, maxlen=None):
        # 遗忘系数，对过去特征的重要性比重
        # 数值越大，对过去依赖越小
        self.alpha = alpha
        self.maxlen = maxlen

    def __call__(self, series):
        if self.maxlen is not None:
            series = series[-self.maxlen:]
        ht = series[0]
        for xt in series[1:]:
            ht = self.alpha * xt + (1-self.alpha) * ht
        return ht

class time_series_double_exponential_compress_representation:

    # 同上

    def __init__(self, alpha, beta, maxlen=None):
        self.alpha = alpha
        self.beta = beta # trend factor
        self.maxlen = maxlen

    def __call__(self, series):
        if self.maxlen is not None:
            series = series[-self.maxlen:]
        if series.size <= 2:
            return series[-1]
        s = series[1]
        b = series[1] - series[0]
        for xt in series[2:]:
            st = self.alpha * xt + (1-self.alpha)(s+b)
            bt = self.beta * (st-s) + (1-self.beta)*b
            # update
            s = st
            b = bt
        ht = st + bt
        return ht

class time_series_feature_successive_compress:

    # i: [1, 1, 2, 2, 3, 0]
    # o: [1, 2, 3, 0]

    def __init__(self, func):
        self.func = func

    