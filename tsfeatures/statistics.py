import pandas as pd
import numpy as np
import tsfresh.feature_extraction.feature_calculators as tsfc

def time_series_absolute_sum_of_changes(series):
    """
    a measurement of time series change

    :return type: float
    """
    return tsfc.absolute_sum_of_changes(series)

def time_series_mean_abs_change(series):
    """
    :return type: float
    """
    return tsfc.mean_abs_change(series)

def time_series_mean_change(series):
    return tsfc.mean_change(series)

def time_series_maximum(series):
    return tsfc.maximum(series)

def time_series_minimum(series):
    return tsfc.minimum(series)

def time_series_mean(series):
    return tsfc.mean(series)

def time_series_median(series):
    return tsfc.median(series)

def time_series_standard_deviation(series):
    return tsfc.standard_deviation(series)

def time_series_variance(series):
    return tsfc.variance(series)

def time_series_kurtosis(series):
    # TODO nan
    return tsfc.kurtosis(series)

def time_series_skewness(series):
    return tsfc.skewness(series)

def time_series_sum_values(series):
    return tsfc.sum_values(series)

def time_series_abs_energy(series: pd.Series):
    """
    why

    :param series:
    :type series:
    :return:
    :return type: float

    example:

    """
    return tsfc.abs_energy(series)

def time_series_sum_of_reoccurring_data_points(series):
    return tsfc.sum_of_reoccurring_data_points(series)

def time_series_sum_of_reoccurring_values(series):
    return tsfc.sum_of_reoccurring_values(series)

def time_series_range(series):
    return time_series_maximum(series) - time_series_minimum(series)

def time_series_count_above_mean(series):
    return tsfc.count_above_mean(series)

def time_series_count_below_mean(series):
    return tsfc.count_below_mean(series)

def time_series_longest_strike_above_mean(series):
    return tsfc.longest_strike_above_mean(series)

def time_series_longest_strike_below_mean(series):
    return tsfc.longest_strike_below_mean(series)

def time_series_mean_second_derivative_central(series):
    return tsfc.mean_second_derivative_central(series)

def time_series_ratio_value_number_to_time_series_length(series):
    return tsfc.ratio_value_number_to_time_series_length(series)

def time_series_percentage_of_reoccurring_values_to_all_values(series):
    return tsfc.percentage_of_reoccurring_values_to_all_values(series)

def time_series_percentage_of_reoccurring_datapoints_to_all_datapoints(series):
    # TODO infinity
    return tsfc.percentage_of_reoccurring_datapoints_to_all_datapoints(series)

def time_series_sample_entropy(series):
    return tsfc.sample_entropy(series)

def time_series_has_duplicate_min(series):
    """
    :return type: boolean
    """
    return tsfc.has_duplicate_min(series)

def time_series_has_duplicate_max(series):
    """
    :return type: boolean
    """
    return tsfc.has_duplicate_max(series)

def time_series_has_duplicate(series):
    """
    :return type: boolean
    """
    return tsfc.has_duplicate(series)

def time_series_variance_larger_than_standard_deviation(series):
    """
    :return type: boolean
    """
    return tsfc.variance_larger_than_standard_deviation(series)

def time_series_sum_of_derivative(series):
    return np.sum(np.diff(series))

def time_series_absolute_sum_of_derivative(series):
    abs_sum = 0
    dseries = np.diff(series).tolist()
    for index in range(len(dseries)-1):
        abs_sum += abs(dseries[index+1]-dseries[index])
    return abs_sum

def time_series_mean_absolute_sum_of_derivative(series):
    abs_sum = time_series_absolute_sum_of_derivative(series)
    return abs_sum / (len(series) - 2)

def time_series_variance_of_derivative(series):
    dseries = pd.Series(np.diff(series))
    return tsfc.variance(dseries)

def time_series_mean_of_derivative(series):
    dseries = pd.Series(np.diff(series))
    return tsfc.mean(dseries)

def time_series_trending_pattern(series):
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
    x = np.array(range(1, len(series)+1))
    y = np.array(series)
    A = np.vstack([x, np.ones(len(x))]).T
    coef, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    eps = np.finfo(np.float32).eps
    if -eps < coef < eps:
        return 0
    else:
        return np.sign(coef)

def time_series_volatility(series):
    """
    wiki: https://en.wikipedia.org/wiki/Volatility_(finance)
    wiki: https://en.wikipedia.org/wiki/Stochastic_volatility
    """

    # TODO

    return 0

def time_series_times_of_large_than_standard_deviation(series):
    return tsfc.standard_deviation(series) / (tsfc.maximum(series) - tsfc.minimum(series))

def time_series_derivative_number_crossing_mean(series):
    dseries = np.diff(series)
    return tsfc.number_crossing_m(dseries, np.mean(dseries))

def time_series_maximum_of_derivative(series):
    return tsfc.maximum(np.diff(series))

def time_series_minimum_of_derivative(series):
    return tsfc.minimum(np.diff(series))




def time_series_agg_autocorrelation_mean(series, lag):
    """
    return as [{"f_agg_mean", float}]
    """
    params = [{"f_agg": "mean", "maxlag": lag}]
    return tsfc.agg_autocorrelation(series, params)[0][-1]

def time_series_agg_autocorrelation_var(series, lag):
    params = [{"f_agg": "var", "maxlag": lag}]
    return tsfc.agg_autocorrelation(series, params)[0][-1]

def time_series_agg_autocorrelation_std(series, lag):
    params = [{"f_agg": "std", "maxlag": lag}]
    return tsfc.agg_autocorrelation(series, params)[0][-1]

def time_series_agg_autocorrelation_median(series, lag):
    params = [{"f_agg": "median", "maxlag": lag}]
    return tsfc.agg_autocorrelation(series, params)[0][-1]

def time_series_approximate_entropy(series, m, r):
    return tsfc.approximate_entropy(series, m, r)

def time_series_autocorrelation(series, lag):
    return tsfc.autocorrelation(series, lag)

def time_series_binned_entropy(series, max_bins):
    return tsfc.binned_entropy(series, max_bins)

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

def time_series_c3(series, lag):
    """
    paper: https://www.macalester.edu/~kaplan/knoxville/PRE05443.pdf

    a measurement of non linearity

    @@

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

    @@

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

def time_series_large_standard_deviation(series, r):
    """
    std(series) > r * (maximum(series) - minimum(series))
    """
    return tsfc.large_standard_deviation(series, r)

def time_series_max_langevin_fixed_point(series: pd.Series, r, m):
    return tsfc.max_langevin_fixed_point(series, r, m)

def time_series_number_crossing_m(series, m):
    """
    stackoverflow: https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python

    a method to detect sign changes
    """
    return tsfc.number_crossing_m(series, m)

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

def time_series_number_cwt_peaks(series, n):
    return tsfc.number_cwt_peaks(series, n)

def time_series_number_peaks(series, n):
    return tsfc.number_peaks(series, n)

def time_series_number_peaks_over_k_standard_deviations(series, k, d):
    """
    :param k: k times standard deviations
    :param d: peak duration
    """
    k_std = k * np.std(series)
    count = 0

    n = 0
    for v in series:
        if v >= k_std:
            n += 1
            if n > d:
                count += 1
        else:
            n = 0
    return count

def time_series_range_count(series, min, max):
    return tsfc.range_count(series, min, max)

def time_series_ratio_beyond_r_sigma(series, r):
    return tsfc.ratio_beyond_r_sigma(series, r)

def time_series_ratio_beyond_1_sigma(series):
    return time_series_ratio_beyond_r_sigma(series, 1)

def time_series_ratio_beyond_2_sigma(series):
    return time_series_ratio_beyond_r_sigma(series, 2)

def time_series_ratio_beyond_3_sigma(series):
    return time_series_ratio_beyond_r_sigma(series, 3)

def time_series_value_count(series, value):
    return tsfc.value_count(series, value)

def time_series_mean_value_count(series):
    mean = np.mean(series)
    return time_series_value_count(series, mean)

def time_series_over_k_sigma_count(series, k):
    """
    :param k: k time sigma
    :return: over k sigma count
    :return type: int
    """
    c = 0
    k_std = k * np.std(series)
    mean = np.mean(series)

    for v in series:
        if not ((-k_std+mean) <= v <= (k_std+mean)):
            c += 1
    return c

def time_series_over_1_sigma_count(series):
    """
    :return: count over 1 sigma 
    :return type: int
    """
    return time_series_over_k_sigma_count(series, 1)

def time_series_over_2_sigma_count(series):
    """
    :return: count over 2 sigma
    :return type: int
    """
    return time_series_over_k_sigma_count(series, 2)

def time_series_over_3_sigma_count(series):
    """
    :return: count over 3 sigma
    :return type: int
    """
    return time_series_over_k_sigma_count(series, 3)

def time_series_ratio_beyond_range_k_sigma(series, k):
    """
    :return: ratio of beyond 1, 2, ..., k sigma series
    :return type: list
    """
    ratios = []

    for t in range(1, k+1):
        ratios.append(time_series_ratio_beyond_r_sigma(series, t))
    return ratios

def time_series_gradient_over_k_sigma_count(series, k):
    dy = np.gradient(series)
    return time_series_over_k_sigma_count(dy, k)

def time_series_gradient_over_1_sigma_count(series):
    return time_series_gradient_over_k_sigma_count(series, 1)

def time_series_gradient_over_2_sigma_count(series):
    return time_series_gradient_over_k_sigma_count(series, 2)

def time_series_gradient_over_3_sigma_count(series):
    return time_series_gradient_over_k_sigma_count(series, 3)

def time_series_gradient_last_over_k_sigma_index(series, k):
    dy = np.gradient(series)
    mean = np.mean(dy)
    k_std = k * np.std(dy)
    for idx, v in enumerate(reversed(dy)):
        if not ((mean-k_std) <= v <= (mean+k_std)):
            break
    else:
        return -1

    return (len(dy) - idx - 1) / time_series_length(series)

def time_series_gradient_last_over_3_sigma_index(series):
    return time_series_gradient_last_over_k_sigma_index(series, 3)

def time_series_gradient_first_over_k_sigma_index(series, k):
    dy = np.gradient(series)
    mean = np.mean(dy)
    k_std = k * np.std(dy)
    for idx, v in enumerate(dy):
        if not ((mean-k_std) <= v <= (mean+k_std)):
            break
    else:
        return -1

    return idx / time_series_length(series)

def time_series_gradient_first_over_3_sigma_index(series):
    return time_series_gradient_first_over_k_sigma_index(series, 3)

def time_series_cid_distance(series1, series2):
    """
    a measurement of two time series.
    paper: http://www.cs.ucr.edu/~eamonn/Complexity-Invariant%20Distance%20Measure.pdf
    """

    ce1 = time_series_cid_ce(series1)
    ce2 = time_series_cid_ce(series2)

    series1 = np.asarray(series1)
    series2 = np.asarray(series2)
    ds = series1 - series2

    d = np.sqrt(np.dot(ds, ds)) * max(ce1, ce2) / min(ce1, ce2)
    return d

def time_series_statistical_features(series):
    """
    features is a vector. hence, **do not change features order**
    new feature add to last location

    dimensions is 64.

    500 size, 64 features
    mean time: 0.157
    std: 0.009
    
    ---
    :series: time series
    :series type: pd.Series
    :return: time series features
    :return type: list
    ---
    """

    features = []

    # measurement of time series changes
    features.append(time_series_absolute_sum_of_changes(series))
    features.append(time_series_mean_abs_change(series))
    features.append(time_series_mean_change(series))
    features.append(time_series_change_rates(series))

    # basis statistics features
    features.append(time_series_maximum(series))
    features.append(time_series_minimum(series))
    features.append(time_series_mean(series))
    features.append(time_series_median(series))
    features.append(time_series_standard_deviation(series))
    features.append(time_series_variance(series))
    features.append(time_series_kurtosis(series))
    features.append(time_series_skewness(series))

    # sum-like features
    features.append(time_series_sum_values(series))
    features.append(time_series_abs_energy(series))
    features.append(time_series_sum_of_reoccurring_data_points(series))
    features.append(time_series_sum_of_reoccurring_values(series))

    # size
    features.append(time_series_range(series))
    features.append(time_series_length(series))
    features.append(time_series_duration(series))

    # about peak-like features
    features.append(time_series_count_above_mean(series))
    features.append(time_series_count_below_mean(series))
    features.append(time_series_longest_strike_above_mean(series))
    features.append(time_series_longest_strike_below_mean(series))
    features.append(time_series_mean_second_derivative_central(series))

    # point-value-like features
    features.append(time_series_ratio_value_number_to_time_series_length(series))
    features.append(time_series_percentage_of_reoccurring_values_to_all_values(series))
    features.append(time_series_percentage_of_reoccurring_datapoints_to_all_datapoints(series))

    # entropy and distance
    features.append(time_series_sample_entropy(series))
    features.append(time_series_cid_ce(series))
    
    # boolean
    features.append(int(time_series_has_duplicate_min(series)))
    features.append(int(time_series_has_duplicate_max(series)))
    features.append(int(time_series_has_duplicate(series)))
    features.append(int(time_series_variance_larger_than_standard_deviation(series)))

    # self other
    features.append(time_series_sum_of_derivative(series))
    features.append(time_series_absolute_sum_of_derivative(series))
    features.append(time_series_mean_absolute_sum_of_derivative(series))
    features.append(time_series_variance_of_derivative(series))
    features.append(time_series_mean_of_derivative(series))
    features.append(time_series_trending_pattern(series))
    features.append(time_series_volatility(series))
    features.append(time_series_times_of_large_than_standard_deviation(series))
    features.append(time_series_derivative_number_crossing_mean(series))

    # self derivative maxmin and location
    features.append(time_series_maximum_of_derivative(series))
    features.append(time_series_minimum_of_derivative(series))
    features.append(time_series_gradient_first_over_3_sigma_index(series))
    features.append(time_series_gradient_last_over_3_sigma_index(series))

    # location of some features
    features.append(time_series_first_location_of_maximum(series))
    features.append(time_series_last_location_of_maximum(series))
    features.append(time_series_first_location_of_minimum(series))
    features.append(time_series_last_location_of_minimum(series))
    features.append(time_series_derivative_first_location_of_maximum(series))
    features.append(time_series_derivative_last_location_of_maximum(series))
    features.append(time_series_derivative_first_location_of_minimum(series))
    features.append(time_series_derivative_last_location_of_minimum(series))

    # sigmalihood features
    features.append(time_series_ratio_beyond_1_sigma(series))
    features.append(time_series_ratio_beyond_2_sigma(series))
    features.append(time_series_ratio_beyond_3_sigma(series))
    features.append(time_series_ratio_beyond_r_sigma(series, 7))
    features.append(time_series_over_1_sigma_count(series))
    features.append(time_series_over_2_sigma_count(series))
    features.append(time_series_over_3_sigma_count(series))
    features.append(time_series_gradient_over_1_sigma_count(series))
    features.append(time_series_gradient_over_2_sigma_count(series))
    features.append(time_series_gradient_over_3_sigma_count(series))
    return features

def time_series_features_mapping(name=None, id=None):
    mapping = OrderedDict([
        ('time_series_absolute_sum_of_changes', 0),
        ('time_series_mean_abs_change', 1),
        ('time_series_mean_change', 2),
        ('time_series_change_rates', 3),
        ('time_series_maximum', 4),
        ('time_series_minimum', 5),
        ('time_series_mean', 6),
        ('time_series_median', 7),
        ('time_series_standard_deviation', 8),
        ('time_series_variance', 9),
        ('time_series_kurtosis', 10),
        ('time_series_skewness', 11),
        ('time_series_sum_values', 12),
        ('time_series_abs_energy', 13),
        ('time_series_sum_of_reoccurring_data_points', 14),
        ('time_series_sum_of_reoccurring_values', 15),
        ('time_series_range', 16),
        ('time_series_length', 17),
        ('time_series_duration', 18),
        ('time_series_count_above_mean', 19),
        ('time_series_count_below_mean', 20),
        ('time_series_longest_strike_above_mean', 21),
        ('time_series_longest_strike_below_mean', 22),
        ('time_series_mean_second_derivative_central', 23),
        ('time_series_ratio_value_number_to_time_series_length', 24),
        ('time_series_percentage_of_reoccurring_values_to_all_values', 25),
        ('time_series_percentage_of_reoccurring_datapoints_to_all_datapoints',26),
        ('time_series_sample_entropy', 27),
        ('time_series_cid_ce', 28),
        ('int(time_series_has_duplicate_min', 29),
        ('int(time_series_has_duplicate_max', 30),
        ('int(time_series_has_duplicate', 31),
        ('int(time_series_variance_larger_than_standard_deviation', 32),
        ('time_series_sum_of_derivative', 33),
        ('time_series_absolute_sum_of_derivative', 34),
        ('time_series_mean_absolute_sum_of_derivative', 35),
        ('time_series_variance_of_derivative', 36),
        ('time_series_mean_of_derivative', 37),
        ('time_series_trending_pattern', 38),
        ('time_series_volatility', 39),
        ('time_series_times_of_large_than_standard_deviation', 40),
        ('time_series_derivative_number_crossing_mean', 41),
        ('time_series_maximum_of_derivative', 42),
        ('time_series_minimum_of_derivative', 43),
        ('time_series_gradient_first_over_3_sigma_index', 44),
        ('time_series_gradient_last_over_3_sigma_index', 45),
        ('time_series_first_location_of_maximum', 46),
        ('time_series_last_location_of_maximum', 47),
        ('time_series_first_location_of_minimum', 48),
        ('time_series_last_location_of_minimum', 49),
        ('time_series_derivative_first_location_of_maximum', 50),
        ('time_series_derivative_last_location_of_maximum', 51),
        ('time_series_derivative_first_location_of_minimum', 52),
        ('time_series_derivative_last_location_of_minimum', 53),
        ('time_series_ratio_beyond_1_sigma', 54),
        ('time_series_ratio_beyond_2_sigma', 55),
        ('time_series_ratio_beyond_3_sigma', 56),
        ('time_series_ratio_beyond_r_sigma', 57),
        ('time_series_over_1_sigma_count', 58),
        ('time_series_over_2_sigma_count', 59),
        ('time_series_over_3_sigma_count', 60),
        ('time_series_gradient_over_1_sigma_count', 61),
        ('time_series_gradient_over_2_sigma_count', 62),
        ('time_series_gradient_over_3_sigma_count', 63)])

    if name is not None:
        return mapping[name]

    if id is not None:
        for name, value in mapping.items():
            if value == id:
                return name

def time_series_statistical_features_with_params(series):

    features = []
    return features

def time_series_features_size():
    series = np.random.randn(1000)
    return len(time_series_statistical_features(series))

s = time_series_features_size()
print(s)
