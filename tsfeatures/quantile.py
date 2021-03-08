import numpy as np
import scipy.stats as stats

# 分位数

class time_series_quantile:
	
	def __init__(self, q):
		self.q = q

	def __call__(self, series):
    		return np.quantile(series, self.q)

def time_series_quantile_features(series):
	features = []
	features.append(time_series_quantile(1/4)(series))
	features.append(time_series_quantile(1/3)(series))
	return features
