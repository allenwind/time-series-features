import numpy as np
import scipy.stats as stats

# 最基本的统计特征

def time_series_maximum(series):
    return np.max(series)

def time_series_minimum(series):
    return np.min(series)



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

def extract_time_series_statistics_features(series):
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
    features.append(time_series_hmean(series))
    features.append(time_series_coefficient_of_variation(series))

    return features
