import numpy

# 时间序列中，特征值的冗余比率

def time_series_ratio_of_duplicate(series):
    # 有冗余
    return (series.size != np.unique(series).size) / series.size

def time_series_ratio_of_duplicate_min(series):
    return (np.sum(series == np.min(series)) >= 2) / series.size

def time_series_ratio_of_duplicate_max(series):
    return (np.sum(series == np.max(series)) >= 2) / series.size

def extract_time_series_duplicate_features(series):
    features = []

    features.append(time_series_ratio_of_duplicate(series))
    features.append(time_series_ratio_of_duplicate_min(series))
    features.append(time_series_ratio_of_duplicate_max(series))
    return features
