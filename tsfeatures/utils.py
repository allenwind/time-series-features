import numpy as np

def check_time_series(series):
    # 验证时间序列是否满足需要的格式

    if not isinstance(series, np.array):
        raise ValueError("only support numpy array")
    
    if len(r.shape) != 1:
        raise ValueError("only support one dimension array now")
