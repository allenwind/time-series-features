from .statistics import time_series_statistical_features
from .classification import time_series_classification_features
from .fitting import time_series_fitting_features

__all__ = []

def time_series_features(series):
    features = []
    features.extend(time_series_statistical_features(series))
    features.extend(time_series_classification_features(series))
    features.extend(time_series_fitting_features(series))
    return features


def compute_features_size(window_size=100):
    series = np.random.uniform(0.1, 10, window_size)
    return len(extract_time_series_features(series))

def extract_time_series_all_features(series):
    features = []

    return features

assert compute_features_size(30) == compute_features_size(50)
