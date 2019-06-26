import pandas as pd
import numpy as np
import tsfresh.feature_extraction.feature_calculators as tsfc

def time_series_sum_of_reoccurring_data_points(series):
    return tsfc.sum_of_reoccurring_data_points(series)

def time_series_sum_of_reoccurring_values(series):
    return tsfc.sum_of_reoccurring_values(series)

def time_series_ratio_value_number_to_time_series_length(series):
    return tsfc.ratio_value_number_to_time_series_length(series)

def time_series_percentage_of_reoccurring_values_to_all_values(series):
    return tsfc.percentage_of_reoccurring_values_to_all_values(series)

def time_series_percentage_of_reoccurring_datapoints_to_all_datapoints(series):
    return tsfc.percentage_of_reoccurring_datapoints_to_all_datapoints(series)

def time_series_has_duplicate_min(series):
    return int(tsfc.has_duplicate_min(series))

def time_series_has_duplicate_max(series):
    return int(tsfc.has_duplicate_max(series))

def time_series_has_duplicate(series):
    return int(tsfc.has_duplicate(series))

def time_series_variance_larger_than_standard_deviation(series):
    return int(tsfc.variance_larger_than_standard_deviation(series))



def time_series_derivative_number_crossing_mean(series):
    dseries = np.diff(series)
    return tsfc.number_crossing_m(dseries, np.mean(dseries))





def time_series_index_mass_quantile(series, q):
    """
    the index of q quantile

    :param q: index i where q of the mass of the series lie left of i
    :type q" float    

    :example:
    >>> index_mass_quantile(pd.Series([1, 2, 1]), [{"q": 0.5}])
    0.6666666666666666
    """
    params = [{"q": q}]
    return tsfc.index_mass_quantile(series, params)[0][-1]



def time_series_large_standard_deviation(series, r):
    """
    std(series) > r * (maximum(series) - minimum(series))
    """
    return tsfc.large_standard_deviation(series, r)

def time_series_max_langevin_fixed_point(series: pd.Series, r, m):
    return tsfc.max_langevin_fixed_point(series, r, m)



def time_series_quantile(series, q):
    """
    wiki: https://en.wikipedia.org/wiki/Quantile
    """
    return tsfc.quantile(series, q)

def time_series_time_reversal_asymmetry_statistic(series, lag):
    """
    paper: Highly comparative feature-based time-series classification
    """
    return tsfc.time_reversal_asymmetry_statistic(series, lag)

