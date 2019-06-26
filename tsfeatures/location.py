import tsfresh.feature_extraction.feature_calculators as tsfc

# 给定特征在时序中的位置, 能够在一定程度上反映时序的周期或异常位置

def time_series_first_location_of_maximum(series):
    return tsfc.first_location_of_maximum(series)

def time_series_last_location_of_maximum(series):
    return tsfc.last_location_of_maximum(series)

def time_series_first_location_of_minimum(series):
    return tsfc.first_location_of_minimum(series)

def time_series_last_location_of_minimum(series):
    return tsfc.last_location_of_minimum(series)

def time_series_derivative_first_location_of_maximum(series):
    return tsfc.first_location_of_maximum(np.diff(series))

def time_series_derivative_last_location_of_maximum(series):
    return tsfc.last_location_of_maximum(np.diff(series))

def time_series_derivative_first_location_of_minimum(series):
    return tsfc.first_location_of_minimum(np.diff(series))

def time_series_derivative_last_location_of_minimum(series):
    return tsfc.last_location_of_minimum(np.diff(series))

def time_series_gradient_last_over_k_sigma_index(series, k):
    # 在使用 k sigma 进行 time series segmentation 时, 
    # 可以使用这个函数

    dy = np.gradient(series)
    mean = np.mean(dy)
    k_std = k * np.std(dy)
    for idx, v in enumerate(reversed(dy)):
        if not ((mean-k_std) <= v <= (mean+k_std)):
            break
    else:
        return -1

    return (len(dy) - idx - 1) / time_series_length(series)

def time_series_gradient_first_over_k_sigma_index(series, k):
    dy = np.gradient(series)
    mean = np.mean(dy)
    k_std = k * np.std(dy)
    for idx, v in enumerate(dy):
        if not ((mean-k_std) <= v <= (mean+k_std)):
            break
    else:
        return -1

    return idx / time_series_length(series)

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
