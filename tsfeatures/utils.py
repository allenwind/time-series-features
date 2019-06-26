
# 冗余性检查
# 相关性检查
# 可视化
# 重要性比较
# 度量时序间的距离

def time_series_cid_distance(series1, series2):
    """
    a measurement of two time series.
    paper:
    http://www.cs.ucr.edu/~eamonn/Complexity-Invariant%20Distance%20Measure.pdf
    """

    ce1 = time_series_cid_ce(series1)
    ce2 = time_series_cid_ce(series2)

    series1 = np.asarray(series1)
    series2 = np.asarray(series2)
    ds = series1 - series2

    d = np.sqrt(np.dot(ds, ds)) * max(ce1, ce2) / min(ce1, ce2)
    return d

def check_time_series_gaussian_noise(series):
    # 高斯白噪声检验
    pass

def find_time_series_max_perid(series):
    pass
