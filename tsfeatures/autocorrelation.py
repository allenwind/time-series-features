import numpy as np
import scipy.stats as stats

# 自相关和时序周期有关的特征

class time_series_autocorrelation:
    
    # 时序的自相关系数
    # wiki:
    # https://en.wikipedia.org/wiki/Autocorrelation#Estimation

    def __init__(self, lag):
        self.lag = lag

    def __call__(self, series):
        v = np.var(series)

        # 如果方差太接近 0 自相关系数没有意义
        if np.isclose(v, 0):
            return np.nan

        m = np.mean(series)
        y1 = series[self.lag:]
        y2 = series[:-self.lag]
        # 可以理解为两个自序列的协方差
        return np.sum((y1 - m) * (y2 - m)) / ((series.size-self.lag) * v)

def time_series_all_autocorrelation(series):
    # 计算所有 lag 的自相关值, 这是一种快速计算的方法，每个 lag 值并非估计值。
    # 计算方法可参考
    # wiki:
    # https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation

    n = series.size
    rs = []
    for lag in range(n):
        r = 0
        # 遍历(0, n-lag)范围
        # 把相隔 lag 的元素相乘求和
        for j in range(n-lag):
            r += series[j+lag] * series[j]
        rs.append(r)
    return rs

class time_series_partial_autocorrelation:

    # 时序的偏自相关系数
    # wiki:
    # https://en.wikipedia.org/wiki/Partial_autocorrelation_function

    def __init__(self, lag):
        self.lag = lag

    def __call__(self, series):
        max_demanded_lag = max([lag["lag"] for lag in param])
        n = len(x)

        # Check if list is too short to make calculations
        if n <= 1:
            pacf_coeffs = [np.nan] * (max_demanded_lag + 1)
        else:
            if (n <= max_demanded_lag):
                max_lag = n - 1
            else:
                max_lag = max_demanded_lag
            pacf_coeffs = list(pacf(x, method="ld", nlags=max_lag))
            pacf_coeffs = pacf_coeffs + [np.nan] * max(0, (max_demanded_lag - max_lag))

        return [("lag_{}".format(lag["lag"]), pacf_coeffs[lag["lag"]]) for lag in param]

def time_series_all_partial_autocorrelation(series):
    # 所有偏自相关系数
    pass

def time_series_all_fft_coefficient(series):
    # 所有 fft 系数
    pass

def time_series_periodic_features(series):
    return  []

def extract_time_series_autocorrelation_based_features(series):
    features = []

    features.extend(time_series_all_autocorrelation(series))
    return features
