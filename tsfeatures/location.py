def time_series_first_location_of_maximum(series):
    return tsfc.first_location_of_maximum(series)

def time_series_last_location_of_maximum(series):
    return tsfc.last_location_of_maximum(series)

def time_series_first_location_of_minimum(series):
    return tsfc.first_location_of_minimum(series)

def time_series_last_location_of_minimum(series):
    return tsfc.last_location_of_minimum(series)

def time_series_derivative_first_location_of_maximum(series):
    return tsfc.first_location_of_maximum(np.diff(series))

def time_series_derivative_last_location_of_maximum(series):
    return tsfc.last_location_of_maximum(np.diff(series))

def time_series_derivative_first_location_of_minimum(series):
    return tsfc.first_location_of_minimum(np.diff(series))

def time_series_derivative_last_location_of_minimum(series):
    return tsfc.last_location_of_minimum(np.diff(series))
