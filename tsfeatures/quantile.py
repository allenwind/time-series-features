
# 分位数

def time_series_quantile(series, q):
	return pd.Series.quantile(series, q)