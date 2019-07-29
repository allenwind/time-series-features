import numpy as np
import scipy.spatial as spatial

# from .special import time_series_cid_ce
# from .autocorrelation import time_series_all_autocorrelation

# TODO:
# 冗余性检查
# 相关性检查
# 可视化
# 重要性比较
# 度量时序间的距离
# dtw 实现

def time_series_cid_distance(series1, series2):
    # a distance measurement of two time series.
    # paper:
    # http://www.cs.ucr.edu/~eamonn/Complexity-Invariant%20Distance%20Measure.pdf

    ce1 = time_series_cid_ce(series1)
    ce2 = time_series_cid_ce(series2)
    ds = series1 - series2

    d = np.sqrt(np.dot(ds, ds)) * max(ce1, ce2) / min(ce1, ce2)
    return d

def time_series_dtw_distance(series1, series2):
    # 简单说, dtw 的基本原理是给定一个适应性窗口, 时序间两个时间点的距离是这个窗口
    # 内的点距离的最小值.
    # http://www.mathcs.emory.edu/~lxiong/cs730_s13/share/slides/searching_sigkdd2012_DTW.pdf
    # https://pdfs.semanticscholar.org/05a2/0cde15e172fc82f32774dd0cf4fe5827cad2.pdf

    pass

def check_time_series(series):
    # 验证时间序列是否满足需要的格式

    if not isinstance(series, np.array):
        raise ValueError("only support numpy array")
    
    if len(r.shape) != 1:
        raise ValueError("only support one dimension array now")

def check_time_series_gaussian_noise(series):
    # 高斯白噪声检验
    pass

def find_time_series_max_periodic(series, offset=1):
    # 如何时序存在周期, 那么自相关函数会呈现明显的规律
    # offset 表示忽略自相关函数中的前 n 个自相关系数
    # 通常来说，较长的序列且复杂的序列会在 lag 较小时
    # 表示较大的自相关性.
    if not offset:
        offset = max(1, len(series)//100)

    auto = time_series_all_autocorrelation(series)
    auto = np.array(auto[offset:])
    return int(np.argmax(auto)) + 1

def cycle_rolling(series, rotate):
    idx = rotate % len(series)
    return np.concatenate([series[-idx:], series[:-idx]])

def mahalanobis_distance(obs, X, center="zero"):
    # mahalanobis distance of obs to X
    # wiki:
    # https://en.wikipedia.org/wiki/Mahalanobis_distance

    # 使用这个方法的参考 paper:
    # Online Anomaly Detection for Hard Disk Drives Based on Mahalanobis Distance
    # 事实上，它可以结合 KNN 使用，提升分类性能

    # 计算协方差矩阵
    cov = np.cov(X.T)

    # 计算数据集的中心
    if center == "zero":
        center = np.zeros(cov.shape[1])
    else:
        center = np.mean(X, axis=0)

    # 矩阵的伪逆
    icov = np.linalg.pinv(cov)
    # 计算 obs 到 center 的 Mahalanobis distance
    d = spatial.distance.mahalanobis(obs, center, icov)
    return d
