# 时间序列的光滑处理

TODO 补充更多解释

simple moving average
$$
\begin{aligned} \overline{p}_{\mathrm{SM}} &=\frac{p_{M}+p_{M-1}+\cdots+p_{M-(n-1)}}{n} \\ &=\frac{1}{n} \sum_{i=0}^{n-1} p_{M-i} \end{aligned}
$$
迭代处理
$$
\overline{p}_{\mathrm{SM}}=\overline{p}_{\mathrm{SM}, \mathrm{prev}}+\frac{1}{n}\left(p_{M}-p_{M-n}\right)
$$


Cumulative moving average
$$
C M A_{n}=\frac{x_{1}+\cdots+x_{n}}{n} \\
C M A_{n+1}=\frac{x_{n+1}+n \cdot C M A_{n}}{n+1}
$$

$$
\begin{aligned} C M A_{n+1} &=\frac{x_{n+1}+n \cdot C M A_{n}}{n+1} \\ &=\frac{x_{n+1}+(n+1-1) \cdot C M A_{n}}{n+1} \\ &=\frac{(n+1) \cdot C M A_{n}+x_{n+1}-C M A_{n}}{n+1} \\ &=C M A_{n}+\frac{x_{n+1}-C M A_{n}}{n+1} \end{aligned}
$$
Weighted moving average
$$
\mathrm{WMA}_{M}=\frac{n p_{M}+(n-1) p_{M-1}+\cdots+2 p_{(M-n+2)}+p_{(M-n+1)}}{n+(n-1)+\cdots+2+1}
$$


Exponential moving average
$$
S_{t}=\left\{\begin{array}{ll}{Y_{1},} & {t=1} \\ {\alpha \cdot Y_{t}+(1-\alpha) \cdot S_{t-1},} & {t>1}\end{array}\right.
$$


Double exponential smoothing
$$
\begin{array}{l}{s_{1}=x_{1}} \\ {b_{1}=x_{1}-x_{0}} \\
\begin{aligned} s_{t} &=\alpha x_{t}+(1-\alpha)\left(s_{t-1}+b_{t-1}\right) \\ b_{t} &=\beta\left(s_{t}-s_{t-1}\right)+(1-\beta) b_{t-1} \end{aligned} \end{array}
$$
for forecasting $F_{t+m}=s_{t}+m b_{t}$



Triple exponential smoothing
