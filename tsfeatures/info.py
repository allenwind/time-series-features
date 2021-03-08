import numpy as np

# 信息论或信号处理(小波)中的有关度量

# entropy:
# 1. approximate entropy
# 2. binned entropy
# 3. sample entropy
# 这三个特征都可以用来度量复杂性, 因此可以用于分类任务.

class time_series_approximate_entropy:

    def __init__(self, m=2, r=0.2):
        self.m = m # 窗口大小
        self.r = r # 阈值过滤等级

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
        return self._phi(series, m, r) - self._phi(series, m+1, r)

    def _phi(self, series, m, r):
        # 转换为滑动窗口形式
        X = self._series2X(series, m)

        # 让所有向量互相作差
        # 扩展维度以便使用 numpy 的 broadcasting
        X1 = X[:, np.newaxis] # (_, 1, m)
        X2 = X[np.newaxis, :] # (1, _, m)
        d = np.abs(X1 - X2) # (_, _, m)

        # 计算 Chebyshev distance
        d = np.max(d, axis=2)

        # 距离小于 r 的计数
        C = np.sum(d <=r , axis=0) / (series.size-m+1)
    
        # 计算 phi 值
        phi = np.sum(np.log(C)) / (series.size-m+1)
        return phi

    def _series2X(self, series, m):
        return np.array([series[i:i+m] for i in range(series.size-m+1)])

class time_series_sample_entropy(time_series_approximate_entropy):

    # 样本熵,度量时序复杂性的一种方式
    # wiki:
    # https://www.wikiwand.com/en/Sample_entropy

    def __call__(self, series):
        if series.size < self.m+1:
            return 0
        
        r = self.r * np.std(series)
        A = self._phi(series, self.m+1, r)
        B = self._phi(series, self.m, r)
        return -np.log(A/B)

    def _phi(self, series, m, r):
        # 转换为滑动窗口形式
        X = self._series2X(series, m)

        # 让所有向量互相作差
        X1 = X[:, np.newaxis]
        X2 = X[np.newaxis, :]
        d = np.abs(X1 - X2)

        # 计算 Chebyshev distance
        d = np.max(d, axis=2)

        # 所有距离小于 r 的计数
        c = np.sum(d <= r)
        return c

class time_series_binned_entropy:

    # 直接进行简单的概率密度估计
    # 根据熵的定义计算结果
    
    def __init__(self, bins=None):
        self.bins = bins

    def __call__(self, series):
        probs, _ = np.histogram(series, bins=self.bins, density=True)
        probs = probs[probs != 0] # 过滤 0, 否则取对数出现 np.inf
        return -np.sum(probs * np.log(probs))

def time_series_abs_energy(series):
    # 信号处理中的一个概念
    # wiki:
    # https://en.wikipedia.org/wiki/Energy_(signal_processing)
    return np.sum(np.square(series))

def time_series_mean_spectral_energy(series):
    # 时间序列的谱能量
    return np.mean(np.square(np.abs(np.fft.fft(series))))

