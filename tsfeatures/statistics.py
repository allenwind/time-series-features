import numpy as np
import scipy.stats as stats

# 最基本的统计特征

def time_series_maximum(series):
    # 最大值
    return np.max(series)

def time_series_minimum(series):
    # 最小值
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
    # 标准差
    return np.std(series)

def time_series_variance(series):
    # 方差
    return np.var(series)

def time_series_kurtosis(series):
    # 四阶矩，度量分布的峰度
    return stats.kurtosis(series)

def time_series_skewness(series):
    # 三阶矩，度量分布的偏度
    return stats.skew(series)

def time_series_length(series):
    # 时序长度
    return len(series)

def time_series_gmean(series):
    # 几何均值
    return stats.gmean(series)

def time_series_hmean(series):
    # 调和均值
    # 不能存在 0 取值元素
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
    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0]
    return list(np.histogram(series, bins=bins)[0] / series.size)

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
