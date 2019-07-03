import numpy as np

from .statistics import extract_time_series_statistics_features
from .autocorrelation import extract_time_series_autocorrelation_based_features
from .fitting import extract_time_series_fitting_features
from .change import extract_time_series_change_features
from .dispersion import extract_time_series_dispersion_features
from .peaks import extract_time_series_peak_features
from .location import extract_time_series_location_features
from .index import (extract_time_series_forecast_features,
                    extract_time_series_regression_features,
                    extract_time_series_anomaly_features,
                    extract_time_series_classification_features)

__all__ = ["extract_time_series_statistics_features", "extract_time_series_autocorrelation_based_features", "extract_time_series_fitting_features",
           "extract_time_series_change_features", "extract_time_series_dispersion_features", "extract_time_series_peak_features", 
           "extract_time_series_location_features", "extract_time_series_all_features", "compute_features_size"]

def extract_time_series_all_features(series):
    features = []

    features.extend(extract_time_series_statistics_features(series))
    features.extend(extract_time_series_autocorrelation_based_features(series))
    features.extend(extract_time_series_fitting_features(series))
    features.extend(extract_time_series_change_features(series))
    features.extend(extract_time_series_dispersion_features(series))
    features.extend(extract_time_series_peak_features(series))
    features.extend(extract_time_series_location_features(series))
    return np.array(features)

def compute_features_size(window_size=100):
    series = np.random.uniform(0.1, 10, window_size)
    return len(extract_time_series_all_features(series))
