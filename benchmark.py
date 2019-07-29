import numpy as np
from tsfeatures import extract_time_series_all_features

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

def _benchmark_fitting():
    pass

if __name__ == "__main__":
    _benchmark_test()
