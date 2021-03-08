# time series features

时间序列中的机器学习任务如磁盘故障预测、时间序列预测、时间序列异常检测、时间序列分类等等都建立在特征工程上。本项目实现时序场景下常用的特征。在机器学习任务中，时间序列特征的使用大致流程如下:

![features-usage-flow](./asset/features-usage-flow.png)




## 特征类型

本项目包括的特征如下：
1. 基本的统计特征 (最大值、最小值、mean、median、方差、标准差、kurtosis、skewness等等)
2. 分位数特征
2. 自相关（autocorrelation）与周期特征
3. 度量时序变化情况的特征
4. 度量时序分散情况的特征
5. 与信息论和信号处理相关的特征(熵 谱 小波分析)
6. 峰值特征
7. 位置特征(某个特征在时间窗口中的位置)
8. 光滑特征(使用平滑方法 MA EMA 或滤波器对原始时序进行处理的结果)
9. 专用特征(某些 paper 上提到的特征, 用于解决某个或某类特定问题)

以上的特征分类方法在理论上并不表明特征是互相独立的，只是一种习惯分类方法。



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



## 使用

克隆项目到`{your_path}`，

```bash
git clone 
```

打开`.bashrc`添加项目路径到`PYTHONPATH`环境变量中，

```.bashrc
export PYTHONPATH={your_path}/time-series-features:$PYTHONPATH
```

然后，

```bash
source ~/.bashrc
```

例子，

```python
import tsfeatures

series = np.random.uniform(size=100)
features = tsfeatures.extract_time_series_statistics_features(series)
print(features)
```



