import numpy as np

# 信息论或信号处理(小波)中的有关度量
# entropy:
# 1. approximate entropy
# 2. binned entropy
# 3. sample entropy

def _phi(m):
    N = x.size
    x_re = np.array([x[i:i+m] for i in range(N - m + 1)])
    C = np.sum(np.max(np.abs(x_re[:, np.newaxis] - x_re[np.newaxis, :]),
                      axis=2) <= r, axis=0) / (N-m+1)
    return np.sum(np.log(C)) / (N - m + 1.0)

def time_series_approximate_entropy(series, m, r):
    # 度量时间序列的波动的不可预测性
    # wiki:
    # https://en.wikipedia.org/wiki/Approximate_entropy
    
    n = len(series)
    r *= np.std(x)
    if n <= m+1:
        return 0
    return np.abs(_phi(m) - _phi(m + 1))

def time_series_binned_entropy(series, max_bins):
    hist, bin_edges = np.histogram(x, bins=max_bins)
    probs = hist / x.size
    return -np.sum(p * np.math.log(p) for p in probs if p != 0)

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

