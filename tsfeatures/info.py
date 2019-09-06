import numpy as np

# 信息论或信号处理(小波)中的有关度量

# entropy:
# 1. approximate entropy
# 2. binned entropy
# 3. sample entropy
# 这三个特征都可以用来度量复杂性, 因此可以用于分类任务.

class time_series_approximate_entropy:

    def __init__(self, m=2, r=3):
        self.m = m
        self.r = r

    def __call__(self, series):
        # 度量时间序列的波动的不可预测性
        # 数值越小,序列的确定性越大
        # wiki:
        # https://en.wikipedia.org/wiki/Approximate_entropy

        # 窗口太大,无法进行合理的切分
        if series.size <= self.m+1:
            return 0

        r = self.r * np.std(series) # 过滤阈值
        m = self.m # 窗口大小
        return np.abs(self._phi(series, m, r) - self._phi(series, m+1, r))

    def _phi(self, series, m, r):
        # 转换为滑动窗口形式
        X = np.array([series[i:i+m] for i in range(series.size-m+1)])
        C = np.sum(np.max(np.abs(X[:, np.newaxis] - X[np.newaxis, :]), axis=2) <= r, axis=0) / (series.size-m+1)
        return np.sum(np.log(C)) / (series.size-m+1)

class time_series_binned_entropy:

    # 直接进行简单的概率密度估计
    # 根据熵的定义计算结果
    
    def __init__(self, bins=None):
        self.bins = bins

    def __call__(self, series):
        probs, _ = np.histogram(series, bins=self.bins) / series.size
        probs = probs[probs != 0] # 过滤 0, 否则取对数出现 np.inf
        return -np.sum(probs * np.log(probs))

class time_series_sample_entropy:

    # 样本熵,度量时序复杂性的一种方式
    # wiki:
    # https://www.wikiwand.com/en/Sample_entropy

    def __init__(self, m, r):
        self.m = m
        self.r = r

    def __call__(self, series):
        if series.size < self.m+1:
            return 0
        
        r = self.r * np.std(series)
        A = self._phi(series, self.m+1, r)
        B = self._phi(series, self.m, r)
        return -np.log(A/B)

    def _phi(self, series, m, r):
        X = np.array([series[i:i+m] for i in range(series.size-m+1)])
        return np.sum(np.max(np.abs(X[:, np.newaxis] - X[np.newaxis, :]), axis=2) < r, axis=0)

def time_series_sample_entropy(series):
    x = np.array(x)

    sample_length = 1 # number of sequential points of the time series
    tolerance = 0.2 * np.std(x) # 0.2 is a common value for r - why?

    n = len(x)
    prev = np.zeros(n)
    curr = np.zeros(n)
    A = np.zeros((1, 1))  # number of matches for m = [1,...,template_length - 1]
    B = np.zeros((1, 1))  # number of matches for m = [1,...,template_length]

    for i in range(n - 1):
        nj = n - i - 1
        ts1 = x[i]
        for jj in range(nj):
            j = jj + i + 1
            if abs(x[j] - ts1) < tolerance:  # distance between two vectors
                curr[jj] = prev[jj] + 1
                temp_ts_length = min(sample_length, curr[jj])
                for m in range(int(temp_ts_length)):
                    A[m] += 1
                    if j < n - 1:
                        B[m] += 1
            else:
                curr[jj] = 0
        for j in range(nj):
            prev[j] = curr[j]

    N = n * (n - 1) / 2
    B = np.vstack(([N], B[0]))

    # sample entropy = -1 * (log (A/B))
    similarity_ratio = A / B
    se = -1 * np.log(similarity_ratio)
    se = np.reshape(se, -1)
    return se[0]

def time_series_abs_energy(series):
    # 信号处理中的一个概念
    # wiki:
    # https://en.wikipedia.org/wiki/Energy_(signal_processing)
    return np.sum(np.square(series))

def time_series_mean_spectral_energy(series):
    # 时间序列的谱能量
    return np.mean(np.square(np.abs(np.fft.fft(series))))

