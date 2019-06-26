
class time_series_moving_average:

    def __init__(self, ws):
        self.ws = ws

    def __call__(self, series):
        pass

def time_series_trending_pattern(series, eps=None):
    """
    doc: https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.linalg.lstsq.html

    :return: 
    0 for no trending pattern, 
    1 for positive trending pattern, 
    -1 for negative trending pattern,
    :return type: int

    :example:
    >>> time_series_trending_pattern([1, 1, 1])
    0
    >>> time_series_trending_pattern([1, 1, 2])
    1.0
    >>> time_series_trending_pattern([1, 1, -2])
    -1.0
    """

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
