import numpy

# 时间序列中，特征值的冗余比率

def time_series_is_duplicate(series):
    return series.size != np.unique(series).size

def time_series_ratio_of_duplicate(series):
    # 有冗余
    return 1 - np.unique(series).size / series.size

class time_series_ratio_of_duplicate_feature:

    def __init__(self, func):
        if func not in (np.min, np.max, np.mean):
            raise ValueError(func, "not support")
        self.func = func

    def __call__(self, series):
        feature = self.func(series)
        count = np.sum(series == feature)
        return count / series.size

time_series_ratio_of_duplicate_min = time_series_ratio_of_duplicate_feature(np.min)

time_series_ratio_of_duplicate_max = time_series_ratio_of_duplicate_feature(np.max)

def extract_time_series_duplicate_features(series):
    features = []

    features.append(time_series_ratio_of_duplicate(series))
    features.append(time_series_ratio_of_duplicate_min(series))
    features.append(time_series_ratio_of_duplicate_max(series))
    return features
