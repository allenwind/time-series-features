import numpy as np
import tsfresh.feature_extraction.feature_calculators as tsfc

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
    # paper: https://www.macalester.edu/~kaplan/knoxville/PRE05443.pdf

    return tsfc.c3(series, lag)

def time_series_cid_ce(series, normalize=True):
    # CID distance
    # paper: http://www.cs.ucr.edu/~eamonn/Complexity-Invariant%20Distance%20Measure.pdf

    return tsfc.cid_ce(series, normalize)

def time_series_time_reversal_asymmetry_statistic(series, lag):
    # paper: 
    # Highly comparative feature-based time-series classification
    
    return tsfc.time_reversal_asymmetry_statistic(series, lag)
