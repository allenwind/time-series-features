from .statistics import extract_time_series_statistics_features
from .autocorrelation import extract_time_series_autocorrelation_based_features
from .fitting import extract_time_series_fitting_features
from .change import extract_time_series_change_features
from .dispersion import extract_time_series_dispersion_features
from .peaks import extract_time_series_peak_features
from .location import extract_time_series_location_features

# 根据不同的使用场景生成不同的特征集
# 目前包括的场景: 预测 回归 异常检测 分类
# 这些特征集需要根据经验调整

def extract_time_series_forecast_features(series):
    features = []

    features.extend(extract_time_series_fitting_features(series))
    features.extend(extract_time_series_autocorrelation_based_features(series))
    features.extend(extract_time_series_statistics_features(series))
    return features

def extract_time_series_regression_features(series):
    features = []

    features.extend(extract_time_series_fitting_features(series))
    return features

def extract_time_series_anomaly_features(series):
    features = []

    features.extend(extract_time_series_statistics_features(series))
    features.extend(extract_time_series_change_features(series))
    features.extend(extract_time_series_classification_features(series))
    features.extend(extract_time_series_peak_features(series))
    return features

def extract_time_series_classification_features(series):
    pass
