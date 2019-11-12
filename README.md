# time series features

在 [time-series-mission](time-series-mission.git) 中我们提到过，机器学习任务都建立在特征工程上。本项目实现时序场景下常用的特征。

## 使用

直接作为某项目的 sub-repo

API 兼容 scikit-learn 项目中, 可以配合 sklearn 上的模型使用.

在机器学习任务中, 特征的使用流程如下:

![features-usage-flow](./asset/features-usage-flow.png)

或者， 根据业务场景，定义需求的特征：


## 特征类型

本项目包括的特征如下:
1. 基本的统计特征 (均值方差等等)
2. 分位数特征
2. 自相关与周期特征
3. 度量时序变化情况的特征
4. 度量时序分散情况的特征
5. 与信息论和信号处理相关的特征(熵 谱 小波分析)
6. 峰值特征
7. 位置特征(某个特征在时间窗口中的位置)
8. 光滑特征(使用平滑方法 MA EMA 或滤波器对原始时序进行处理的结果)
9. 专用特征(某些 paper 上提到的特征, 用于解决某个或某类特定问题)

以上的分类并不表明特征是互相独立的.

更多关于以上特征的说明参考(Latex 可能无法渲染):

1. [时间序列特征说明](./md/time-series-features.md)
2. [时间序列平滑处理说明](./md/time-series-smoothing.md)


特征工程

1. 普通特征 (统计特征、拟合特征、分类特征)
2. dtw & wavelet
3. autocorrelation
4. reprecentation learning (深度学习自动特征提取)

目前运用到预测中的特征包括：最大值、最小值、mean、median、方差、标准差、kurtosis、skewness 等等

更丰富的时序特征提取模块见时间序列特征提取, 该模块包括了特征重要性评估

1~3 的特征提取通过滑动窗口的方式进行,滑动步长通常为 1. 4 为深度学习中的自动特征提取方法, 通过我们会使用 CNN 提取局部特征, LSTM 提取长期依赖特征.下面会展开这方面方法.

特征重要性评估


## 特征索引

根据时间序列的使用场景, 我们对特征进行索引, 目前包括的使用场景:

1. 预测(forecasting)
2. regression
3. anomaly detection (changepoints, outliers)
4. classification

## 特征重要性

以上说了如此多特征, 哪些特征更重要?哪些不重要?

关于模型的先天"优越性"我们想到"没有免费午餐定理", 类似地, 特征也没有先天"优越性", 什么特征好或不好, 视使用场景而定.

因此, 我们需要根据场景来选择我们需要的特征, 更直接的做法是在特征提取后做特征过滤.

## 数据与特征

在机器学习任务中，特征的使用伴随着数据探索的整个过程，可以说，使用什么特征等价于我们如何理解数据和机器学习任务。

## 实现模板

不带参数的特征：

```python
def time_series_xx(series):
    # 无参数的特征提取器实现模板
    # 返回类型为 float 或 list
    # 使用方法:
    # feature = time_series_xx(series)
    pass
```

带参数的特征：

```python
class time_series_xx:

    # 有参数的特征提取器实现模板
    # 可以使用类继承方法实现
    # 返回类型为 float、int 或 list
    # 使用方法
    # feature = time_series_xx(params)(series)
    
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def __call__(self, series):
        pass

```

获取 `np.array` 类型的长度使用 `size` 属性，而不是 `len`.

## 使用例子

为了更好地理解如何使用这个特征库，这里演示几个例子。

例子：
1. 时序预测
2. 时序异常检测
3. 时序分类

## 相关 paper

更多相关的说明请看项目源码中的注释.

## TODO

1. 补充特征计算文档
2. 添加测试代码
