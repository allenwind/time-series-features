import numpy as np

from .special import time_series_cid_ce
from .autocorrelation import time_series_all_autocorrelation

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
    if not isinstance(series, np.array):
        raise ValueError("only support numpy array")
    
    if len(r.shape) != 1:
        raise ValueError("only support one dimension array now")

def check_time_series_gaussian_noise(series):
    # 高斯白噪声检验
    pass

def find_time_series_max_periodic(series):
    auto = time_series_all_autocorrelation(series)
    auto = np.array(auto[1:])
    return np.argmax(auto)
