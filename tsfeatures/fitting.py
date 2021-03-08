import numpy as np

# 光滑, 去噪与趋势

EPS = 1.1920929e-02

class time_series_moving_average:

    # 移动平均
    # 目前实现方式并非最高效，待优化

    def __init__(self, ws):
        self.ws = ws

    def __call__(self, series):
        hts = []
        for i in range(series.size-self.ws+1):
            ht = np.mean(series[i:i+self.ws])
            hts.append(ht)
        return hts

class time_series_weighted_moving_average:

    # 加权的移动平均

    def __init__(self, weights):
        self.weights = weights
        self.totals = np.sum(weights)
        self.size = len(weights)

    def __call__(self, series):
        hts = []
        for i in range(series.size-self.size+1):
            ht = np.dot(series[i:i+self.size], self.weights) / self.totals
            hts.append(ht)
        return hts

class time_series_exponential_moving_average:

    def __init__(self, alpha=0.95):
        self.alpha = alpha

    def __call__(self, series):
        hts = [series[0]]
        ht = series[0]
        for xt in series[1:]:
            ht = self.alpha * xt + (1-self.alpha) * ht
            hts.append(ht)
        return hts

class time_series_double_exponential_moving_average:
    
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta # trend factor

    def __call__(self, series):
        if series.size == 1:
            return series[-1]
        s = series[1]
        b = series[1] - series[0]
        hts = [a+b]
        for xt in series[2:]:
            st = self.alpha * xt + (1-self.alpha)(s+b)
            bt = self.beta * (st-s) + (1-self.beta)*b
            # update
            s = st
            b = bt
            hts.append(st+bt)
        return hts

class time_series_trending_pattern:

    def __init__(self, eps=None):
        self.eps = eps if eps else EPS

    def __call__(self, series):
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
    features.extend(time_series_exponential_moving_average(0.95)(series))
    return features
