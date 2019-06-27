import tsfresh.feature_extraction.feature_calculators as tsfc

# 和 peak 有关的特征, 通常用在时间序列分类和异常检测中

def time_series_number_cwt_peaks(series, n):
    return tsfc.number_cwt_peaks(series, n)

def time_series_number_peaks(series, n):
    return tsfc.number_peaks(series, n)

def time_series_number_peaks_over_k_standard_deviations(series, k, d):
    """
    :param k: k times standard deviations
    :param d: peak duration
    """
    k_std = k * np.std(series)
    count = 0

    n = 0
    for v in series:
        if v >= k_std:
            n += 1
            if n > d:
                count += 1
        else:
            n = 0
    return count

def extract_time_series_peak_features(series):
    features = []

    return features
