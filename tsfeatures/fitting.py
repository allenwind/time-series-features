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

    def __init__(self, alpha=0.95):
        self.alpha = alpha

    def __call__(self, series):
        values = []
        x0 = series[0]
        values.append(x0)
        for v in series[1:]:
            xt = self.alpha*v + (1-self.alpha)*x0
            x0 = xt
            values.append(xt)
        return values

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

def time_series_moving_average(series):
    values = []
    for w in range(1, 50, 5):
        value = np.mean(series[-w:])
        values.append(value)
    return list(np.array(values) - series[-1])

def time_series_weighted_moving_average(series):
    temp_list = []
    for w in range(1, min(50, DEFAULT_WINDOW), 5):
        w = min(len(x), w)
        coefficient = np.array(range(1, w + 1))
        temp_list.append((np.dot(coefficient, x[-w:])) / float(w * (w + 1) / 2))
    return list(np.array(temp_list) - x[-1])

def time_series_exponential_weighted_moving_average(series):
    temp_list = []
    for j in range(1, 10):
        alpha = j / 10.0
        s = [x[0]]
        for i in range(1, len(x)):
            temp = alpha * x[i] + (1 - alpha) * s[-1]
            s.append(temp)
        temp_list.append(s[-1] - x[-1])
    return temp_list

def time_series_double_exponential_weighted_moving_average(series):
    temp_list = []
    for j1 in range(1, 10, 2):
        for j2 in range(1, 10, 2):
            alpha = j1 / 10.0
            gamma = j2 / 10.0
            s = [x[0]]
            b = [(x[3] - x[0]) / 3]  # s is the smoothing part, b is the trend part
            for i in range(1, len(x)):
                temp1 = alpha * x[i] + (1 - alpha) * (s[-1] + b[-1])
                s.append(temp1)
                temp2 = gamma * (s[-1] - s[-2]) + (1 - gamma) * b[-1]
                b.append(temp2)
            temp_list.append(s[-1] - x[-1])
    return temp_list

def extract_time_series_fitting_features(series):
    # 目前只使用 EMA
    features = []

    features.extend(time_series_exponential_moving_average(0.95)(series))
    return features
