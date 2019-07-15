import numpy as np
import scipy.stats as stats
import tsfresh.feature_extraction.feature_calculators as tsfc

# 信息论或信号处理(小波)中的有关度量

def time_series_approximate_entropy(series, m, r):
    # 度量时间序列的波动的不可预测性
    return tsfc.approximate_entropy(series, m, r)

def time_series_binned_entropy(series, max_bins):
    return tsfc.binned_entropy(series, max_bins)

def time_series_sample_entropy(series):
    return tsfc.sample_entropy(series)

def time_series_abs_energy(series):
    # 信号处理中的一个概念
    # wiki:
    # https://en.wikipedia.org/wiki/Energy_(signal_processing)
    return np.sum(np.square(series))

def time_series_mean_spectral_energy(series):
    # 时间序列的谱能量
    return np.mean(np.square(np.abs(np.fft.fft(series))))

def time_series_binned_entropy(series):
    max_bins = [2, 4, 6, 8, 10, 20]
    values = []
    for value in max_bins:
        values.append(tsfc.binned_entropy(series, value))
    return values
