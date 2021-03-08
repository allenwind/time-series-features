import time
import sys
from functools import wraps
import numpy as np
from tsfeatures import extract_time_series_all_features

def timethis(label, timer=time.time, output=sys.stdout, callback=None):
    """
    带参数的计时器，支持指定计时器类型
    """
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = timer()
            result = func(*args, **kwargs)
            end = timer()
            elapsed = end - start
            if callback is not None:
                callback(elapsed)
            template = "{}\n{}.{} elapsed time:\n{:.3f}s"
            value = template.format(label, func.__module__, func.__name__, elapsed)
            print(value, end="\n\n", file=output)
            return result
        return wrapper
    return decorate

@timethis("time series benchmark")
def _benchmark_test():
    import time

    stats = []
    window_size = 300
    series_size = 1000
    retry = 5
    for _ in range(retry):
        start = time.time()
        for _ in range(series_size-window_size):
            r = np.random.normal(size=window_size)
            _ = extract_time_series_all_features(r)
        end = time.time()
        stats.append(end-start)

    print(stats)

if __name__ == "__main__":
    _benchmark_test()
