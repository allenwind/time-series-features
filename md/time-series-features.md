# 时间序列的特征工程

TODO 补充更多解释

概率分布是随机变量的完整刻画，一旦知道随机变量的概率分布表达式就可以完整地导出该随机变量的数学性质，然而现实情况是我们很难从采样数据中推断随机变量的分布，为此，我们可以通过构造一个采样样本的函数来刻画该分布的一些特征。

## 基本统计特征

Mean
$$
\mathrm{E}[X]=\sum_{i=1}^{k} x_{i} p_{i}=x_{1} p_{1}+x_{2} p_{2}+\cdots+x_{k} p_{k}
$$
Median
$$
\mathrm{P}(X \leq m) \geq \frac{1}{2} \text { and } \mathrm{P}(X \geq m) \geq \frac{1}{2}
$$
Mode

Geometric mean
$$
\left(\prod_{i=1}^{n} x_{i}\right)^{\frac{1}{n}}=\sqrt[n]{x_{1} x_{2} \cdots x_{n}}
$$


Harmonic mean
$$
H=\frac{n}{\frac{1}{x_{1}}+\frac{1}{x_{2}}+\cdots+\frac{1}{x_{n}}}=\frac{n}{\sum_{i=1}^{n} \frac{1}{x_{i}}}=\left(\frac{\sum_{i=1}^{n} x_{i}^{-1}}{n}\right)^{-1}
$$


Variance
$$
\begin{aligned} \operatorname{Var}(X) &=\mathrm{E}\left[(X-\mathrm{E}[X])^{2}\right] \\ &=\mathrm{E}\left[X^{2}-2 X \mathrm{E}[X]+\mathrm{E}[X]^{2}\right] \\ &=\mathrm{E}\left[X^{2}\right]-2 \mathrm{E}[X] \mathrm{E}[X]+\mathrm{E}[X]^{2} \\ &=\mathrm{E}\left[X^{2}\right]-\mathrm{E}[X]^{2} \end{aligned}
$$
standard deviation
$$
s=\sqrt{\frac{1}{N-1} \sum_{i=1}^{N}\left(x_{i}-\overline{x}\right)^{2}}
$$
Skewness
$$
\gamma_{1}=\mathrm{E}\left[\left(\frac{X-\mu}{\sigma}\right)^{3}\right]=\frac{\mu_{3}}{\sigma^{3}}=\frac{\mathrm{E}\left[(X-\mu)^{3}\right]}{\left(\mathrm{E}\left[(X-\mu)^{2}\right]\right)^{3 / 2}}=\frac{\kappa_{3}}{\kappa_{2}^{3 / 2}}
$$
kurtosis
$$
\operatorname{Kurt}[X]=\mathrm{E}\left[\left(\frac{X-\mu}{\sigma}\right)^{4}\right]=\frac{\mu_{4}}{\sigma^{4}}=\frac{\mathrm{E}\left[(X-\mu)^{4}\right]}{\left(\mathrm{E}\left[(X-\mu)^{2}\right]\right)^{2}}
$$



Coefficient of variation
$$
c_{\mathrm{v}}=\frac{\sigma}{\mu}
$$


## 度量分散的特征

standard deviation
$$
s=\sqrt{\frac{1}{N-1} \sum_{i=1}^{N}\left(x_{i}-\overline{x}\right)^{2}}Range
$$

Range
$$
r = max(x) - min(x)
$$
Mean absolute deviation around a central point
$$
\frac{1}{n} \sum_{i=1}^{n}\left|x_{i}-m(X)\right|
$$
m 包括 mean median mode

Median absolute deviation around a central point
$$
median(\left|x_{i}-m(X)\right|)
$$



absolute sum of changes
$$
\sum_{i=1, \ldots, n-1}\left|x_{i+1}-x_{i}\right|
$$

mean abs change
$$
\frac{1}{n} \sum_{i=1, \ldots, n-1}\left|x_{i+1}-x_{i}\right|
$$


mean change
$$
\frac{1}{n} \sum_{i=1, \ldots, n-1} x_{i+1}-x_{i}
$$


mean second derivative central
$$
\frac{1}{n} \sum_{i=1, \ldots, n-1} \frac{1}{2}\left(x_{i+2}-2 \cdot x_{i+1}+x_{i}\right)
$$



## 度量变化的特征

mean second derivative central
$$
f^{\prime \prime}(x) \approx \frac{\delta_{h}^{2}[f](x)}{h^{2}}=\frac{\frac{f(x+h)-f(x)}{h}-\frac{f(x)-f(x-h)}{h}}{h}=\frac{f(x+h)-2 f(x)+f(x-h)}{h^{2}}
$$






## 度量自相关和周期的特征

autocorrelation
$$
R(l)=\frac{1}{(n-l) \sigma^{2}} \sum_{t=1}^{n-l}\left(X_{t}-\mu\right)\left(X_{t+l}-\mu\right)
$$
l is lag

partial autocorrelation


$$
\alpha_{k}=\frac{\operatorname{Cov}\left(x_{t}, x_{t-k} | x_{t-1}, \ldots, x_{t-k+1}\right)}{\sqrt{\operatorname{Var}\left(x_{t} | x_{t-1}, \ldots, x_{t-k+1}\right) \operatorname{Var}\left(x_{t-k} | x_{t-1}, \ldots, x_{t-k+1}\right)}}
$$


## 信息度量

abs energy
$$
E=\sum_{i=1, \ldots, n} x_{i}^{2}
$$
Approximate entropy

Sample entropy

binned entropy
$$
\sum_{k=0}^{\min (\max -b i n s, l e n(x))} - p_{k} \log \left(p_{k}\right) \cdot \mathbf{1}_{\left(p_{k}>0\right)}
$$



 cumulative sum
$$
\begin{array}{l}{S_{0}=0} \\ {S_{n+1}=\max \left(0, S_{n}+x_{n}-\omega_{n}\right)}\end{array}
$$

## 分类特征

Discrimination power of measures for nonlinearity, c3
$$
\frac{1}{n-2 \operatorname{lag}} \sum_{i=0}^{n-2 l a g} x_{i+2 \cdot \operatorname{lag}}^{2} \cdot x_{i+\operatorname{lag}} \cdot x_{i}
$$


an estimate for a time series complexity, cid_ce
$$
\sqrt{\sum_{i=0}^{n-2 l a g}\left(x_{i}-x_{i+1}\right)^{2}}
$$

time reversal asymmetry statistic
$$
\frac{1}{n-2 l a g} \sum_{i=0}^{n-2 l a g} x_{i+2 \cdot l a g}^{2} \cdot x_{i+l a g}-x_{i+l a g} \cdot x_{i}^{2}
$$
Highly comparative feature-based time-series classification
