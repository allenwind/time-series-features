import numpy as np
from .utils import cycle_rolling

# 一些高级特征, 来自某些 paper 或专用与某个场景的特征
# 实现中已经注明 papers.

def time_series_sum(series):
    # 时序的累加
    return np.sum(series)

def time_series_abs_sum(series):
    # 时序改变的累加
    return np.sum(np.abs(series))

def time_series_c3(series, lag):
    # a measurement of non linearity
    # paper:
    # https://www.macalester.edu/~kaplan/knoxville/PRE05443.pdf

    n = len(series)
    if 2 * lag >= n:
        return 0
    else:
        return np.mean((cycle_rolling(x, 2 * -lag) * cycle_rolling(x, -lag) * x)[0:(n - 2 * lag)])

def time_series_cid_ce(series):
    # CID distance
    # paper:
    # http://www.cs.ucr.edu/~eamonn/Complexity-Invariant%20Distance%20Measure.pdf

    d = np.diff(series)
    return np.sqrt(np.dot(d, d))

def time_series_time_reversal_asymmetry_statistic(series, lag):
    # paper: 
    # Highly comparative feature-based time-series classification
    
    n = len(series)
    if 2 * lag >= n:
        return 0
    else:
        one_lag = cycle_rolling(series, -lag)
        two_lag = cycle_rolling(series, 2 * -lag)
        return np.mean((two_lag * two_lag * one_lag - one_lag * series * series)[0:(n - 2 * lag)])
