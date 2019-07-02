import numpy as np

# 光滑, 去噪与趋势

EPS = 1.1920929e-02

class time_series_moving_average:

    # 移动平均

    def __init__(self, ws):
        self.ws = ws

    def __call__(self, series):
        pass

class time_series_weighted_moving_average:

    # 加权的移动平均

    def __init__(self, weights):
        self.weights = weights

    def __call__(self, series):
        pass

class time_series_exponential_moving_average:

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, series):
        pass

class time_series_double_exponential_moving_average:

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, series):
        pass

def time_series_trending_pattern(series, eps=EPS):
    # 趋势模式

    if eps is None:
        eps = np.finfo(np.float32).eps

    x = np.array(range(1, len(series)+1))
    y = np.array(series)
    A = np.vstack([x, np.ones(len(x))]).T
    coef, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    if -eps < coef < eps:
        return 0
    else:
        return np.sign(coef)

def extract_time_series_fitting_features(series):
    # 目前只使用 EMA
    features = []

    # features.extend(time_series_exponential_moving_average(0.99)(series))
    return features
