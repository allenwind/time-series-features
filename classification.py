import tsfresh.feature_extraction.feature_calculators as tsfc

"""
see:
https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html
"""

def time_series_agg_autocorrelation(series, param):
    return tsfc.agg_autocorrelation(series, param)

def time_series_partial_autocorrelation(series, param):
    """
    wiki: https://newonlinecourses.science.psu.edu/stat510/node/62/

    :series type: pd.Series
    :param type: int
    """
    return tsfc.partial_autocorrelation(series, param)

def time_series_classification_features(series):
    return []
