from tsfeatures import compute_features_size

def _benchmark_test():
    import time

    stats = []
    window_size = 300
    series_size = 1000
    retry = 5
    for _ in range(retry):
        start = time.time()
        for _ in range(series_size-window_size):
            _ = compute_features_size(window_size)
        end = time.time()
        stats.append(end-start)

    print(stats)

def _benchmark_fitting():
    pass

if __name__ == "__main__":
    _benchmark_test()
