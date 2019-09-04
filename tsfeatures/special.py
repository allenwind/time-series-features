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
        if 2 * self.lag >= n:
            return 0
        else:
            return np.mean((np.roll(x, 2 * -self.lag) * np.roll(x, -self.lag) * x)[0:(n - 2 * self.lag)])

class time_series_time_reversal_asymmetry_statistic:

    def __init__(self, lag):
        self.lag = lag

    def ___call__(self, series):
        # paper: 
        # Highly comparative feature-based time-series classification
        
        n = len(series)
        if 2 * self.lag >= n:
            return 0
        else:
            one_lag = np.roll(series, -self.lag)
            two_lag = np.roll(series, 2 * -self.lag)
            return np.mean((two_lag * two_lag * one_lag - one_lag * series * series)[0:(n - 2 * self.lag)])
