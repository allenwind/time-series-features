"""时间序列特征提取实现模板

该模块实现时间序列的特征提取, 包括统计特征(中心 变化 分散 周期) 拟合特征(光滑处理)
以及和分类相关的特征. 实现这些特征的基础模块包括: numpy scipy 以及 tsfresh 尽量
不要使用 tsfresh, 其计算性能慢. 注意:
    1. 特征的维度和 window_size 是否有关.
    2. 特征计算返回 float int 类型还是 list 类型.
"""

def time_series_xx(series):
    # 无参数的特征提取器实现模板
    # 返回类型为 float 或 list
    # 使用方法:
    # feature = time_series_xx(series)
    pass

class time_series_xx:

    # 有参数的特征提取器实现模板
    # 可以使用类继承方法实现
    # 返回类型为 float 或 list
    # 使用方法
    # feature = time_series_xx(params)(series)
    
    def __init__(self, params):
        self.params = params

    def __call__(self, series):
        pass

