import numpy as np
import scipy.stats as stats
import tsfresh.feature_extraction.feature_calculators as tsfc

# 度量时序改变的特征

def time_series_mean_change(series):
    # 度量时序的变化
    return np.mean(np.diff(series))

def time_series_absolute_mean_of_changes(series):
    # 度量时序的变化
    return np.mean(np.abs(np.diff(series)))

def time_series_maximum_of_derivative(series):
    return tsfc.maximum(np.diff(series))

def time_series_minimum_of_derivative(series):
    return tsfc.minimum(np.diff(series))

def time_series_mean_second_derivative_central(series):
    # 度量时序的变化
    # wiki:
    # https://en.wikipedia.org/wiki/Finite_difference#Higher-order_differences
    
    return tsfc.mean_second_derivative_central(series)

class time_series_zero_crossing:

    # 和特征 time_series_derivative_number_crossing_mean
    # 类似, 但前者默认 threshold = 0
    
    def __init__(self, threshold=0):
        self.threshold = threshold

    def __call__(self, series):
        sign = np.heaviside(series[1:], 0)
        return np.sum(sign * np.abs(np.diff(series)) >= self.threshold)

def time_series_number_crossing_m(series, m):
    # a method to detect sign changes
    # stackoverflow: 
    # https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python

    return tsfc.number_crossing_m(series, m)

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

def time_series_derivative_number_crossing_mean(series):
    d = np.diff(series)
    return tsfc.number_crossing_m(d, np.mean(d))

def extract_time_series_change_features(series):
    features = []

    return features
