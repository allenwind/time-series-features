
# 一些高级特征, 来自某些 paper 或专用与某个场景的特征

def time_series_cumsum(series):
    # 时序的累加
    return np.cumsum(series)

def time_series_abs_cumsum(series):
    # 时序改变的累加
    return np.cumsum(np.abs(series))



def time_series_value_distribution(series):
    # 数值需要变换到 0, 1 区间上
    thresholds = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0, 1.0]
    return list(np.histogram(series, bins=thresholds)[0] / float(len(series)))


def time_series_c3(series, lag):
    """
    paper: https://www.macalester.edu/~kaplan/knoxville/PRE05443.pdf

    a measurement of non linearity

    :param series: time series
    :type series: pd.Series

    :param lag: lag
    :type lag: int

    :return: c3 features
    :return type: float

    :example:
    >>> time_series_c3(pd.Series([1, 2, 1]), lag=1)
    2.0
    """
    return tsfc.c3(series, lag)

def time_series_cid_ce(series, normalize=True):
    """
    paper: http://www.cs.ucr.edu/~eamonn/Complexity-Invariant%20Distance%20Measure.pdf

    a measurement of CID distance

    :param series: time series
    :type series: pd.Series

    :param normalize: normalize input time series
    :param type: boolean

    :return: cid distance
    :return type: float
    
    :example:
    >>> cid_ce(pd.Series([1, 2, 1]), True)
    2.9999999999999996
    >>> cid_ce(pd.Series([1, 2, 1]), False)
    1.4142135623730951
    """

    return tsfc.cid_ce(series, normalize)

