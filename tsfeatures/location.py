import numpy as np

# 给定特征在时序中的位置, 能够在一定程度上反映时序的周期或异常位置
# 这些位置信息可以与 change 和 dispersion 特征配合使用.
# 例如我们想度量一个时间窗口内, 时序的最大增量, 我们需要知道,
# 时序的最大值, 最小值, 首次最大值位置, 末次最小值未知.

def time_series_last_value(series):
    # 序列最后值
    return series[-1]

def time_series_first_value(series):
    # 序列首值
    return series[0]

def time_series_first_location_of_maximum(series):
    return np.argmax(series)

def time_series_last_location_of_maximum(series):
    return  len(series) - np.argmax(series[::-1])

def time_series_first_location_of_minimum(series):
    return np.argmin(series)

def time_series_last_location_of_minimum(series):
    return len(series) - np.argmin(series[::-1])

def time_series_first_location_of_nonzero(series):
    # 第一个非零值索引
    return np.where(series==0)[0][0]

def time_series_last_location_of_nonzero(series):
    # 最后一个非零值索引
    return np.where(series==0)[0][-1]

def time_series_derivative_first_location_of_maximum(series):
    return time_series_first_location_of_maximum(np.diff(series))

def time_series_derivative_last_location_of_maximum(series):
    return time_series_last_location_of_maximum(np.diff(series))

def time_series_derivative_first_location_of_minimum(series):
    return time_series_first_location_of_minimum(np.diff(series))

def time_series_derivative_last_location_of_minimum(series):
    return time_series_last_location_of_minimum(np.diff(series))

class time_series_gradient_last_over_k_sigma_index:

    def __init__(self, k):
        self.k = k

    def __call__(self, series):
        # 在使用 k sigma 进行 time series segmentation 时, 
        # 可以使用这个函数. 当 series 服从正太分布式时, k 取 3 足够了.
        # 这个函数不会加入 location features 中.

        dy = np.gradient(series)
        mean = np.mean(dy)
        k_std = self.k * np.std(dy)
        for idx, v in enumerate(reversed(dy)):
            if not ((mean-k_std) <= v <= (mean+k_std)):
                break
        else:
            return -1

        return (len(dy) - idx - 1) / len(series)

class time_series_gradient_first_over_k_sigma_index:

    def __init__(self, k):
        self.k = k

    def __call__(self, series):
        dy = np.gradient(series)
        mean = np.mean(dy)
        k_std = self.k * np.std(dy)
        for idx, v in enumerate(dy):
            if not ((mean-k_std) <= v <= (mean+k_std)):
                break
        else:
            return -1

        return idx / len(series)

def extract_time_series_location_features(series):
    features = []
    features.append(time_series_first_location_of_maximum(series))
    features.append(time_series_last_location_of_maximum(series))
    features.append(time_series_first_location_of_minimum(series))
    features.append(time_series_last_location_of_minimum(series))
    features.append(time_series_derivative_first_location_of_maximum(series))
    features.append(time_series_derivative_last_location_of_maximum(series))
    features.append(time_series_derivative_first_location_of_minimum(series))
    features.append(time_series_derivative_last_location_of_minimum(series))
    return features
