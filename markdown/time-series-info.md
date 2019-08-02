信息论角度看时间序列 representation

熵的计算方法

$$
H(X)=-\sum_{i=1}^{n} p\left(x_{i}\right) \log p\left(x_{i}\right)
$$

但是时间序列没有具体给出序列的分布，无法直接计算。但可以估计，具体的估计方法有三种：

1. binned entropy
2. approximate entropy
3. sample entropy


