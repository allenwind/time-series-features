from .statistics import time_series_statistical_features
from .classification import time_series_classification_features
from .fitting import time_series_fitting_features

__all__ = ["time_series_statistical_features", 
           "time_series_classification_features",
           "time_series_fitting_features",
           "time_series_features"]

def time_series_features(series):
    features = []
    features.extend(time_series_statistical_features(series))
    features.extend(time_series_classification_features(series))
    features.extend(time_series_fitting_features(series))
    return features
