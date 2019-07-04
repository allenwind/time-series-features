import numpy as np
import scipy.stats as stats

# 最基本的统计特征

def time_series_maximum(series):
    return np.max(series)

def time_series_minimum(series):
    return np.min(series)

def time_series_mean(series):
    # 均值
    return np.mean(series)

def time_series_median(series):
    # 中位数
    return np.median(series)

def time_series_mode(series):
    # 众数
    return stats.mode(series).mode[0]

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
    # 几何均值
    return stats.gmean(series)

def time_series_hmean(series):
    # 调和均值
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

def time_series_value_distribution(series):
    # 时间序列的经验分布
    # 数值需要变换到 0, 1 区间上
    # 自定义区间的数值统计分布
    thresholds = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0]
    return list(np.histogram(series, bins=thresholds)[0] / float(len(series)))

def extract_time_series_statistics_features(series):
    features = []

    # 基本统计特征
    features.append(time_series_minimum(series))
    features.append(time_series_maximum(series))
    features.append(time_series_mean(series))
    features.append(time_series_median(series))
    features.append(time_series_mode(series))
    features.append(time_series_standard_deviation(series))
    features.append(time_series_variance(series))
    features.append(time_series_kurtosis(series))
    features.append(time_series_skewness(series))
    features.append(time_series_length(series))

    # 几何均值和调和均值不直接引入
    # features.append(time_series_gmean(series))
    # features.append(time_series_hmean(series))
    features.append(time_series_coefficient_of_variation(series))
    features.extend(time_series_value_distribution(series))
    return features
