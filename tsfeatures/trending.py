import numpy as np

# 时间序列趋势特征

def time_series_linear_regression_coefficient(series):
    pass

def time_series_stock_price_momentum(series):
    """https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html"""
    pass

def time_series_positive_growth(series):
    length = len(series)
    coeffs = np.polyfit(np.arange(length), series, 1)
    return float(coeffs[-2])

def time_series_history_retracement(series):
    """最大历史回撤"""
    ms = []
    m = -np.inf
    for i in series:
        if i > m:
            m = i
        ms.append(m)
    ms = np.array(ms)
    return (series - ms) / ms

def time_series_linear_trending_mu_sigma(series):
    # 线性趋势中，时序的均值和方差
    pass

def time_series_sequences_and_reversals_test(series):
    pass

class time_series_trending_type:

    def __init__(self, trans_pairs):
        self.trans_pairs = trans_pairs

    def __call__(self, series):
        pass
