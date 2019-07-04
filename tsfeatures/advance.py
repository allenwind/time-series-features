
# TODO@深度学习的特征提取
# 包括自编码器和变分自编码器
# 其基本思路是利用解码器的重构误差来判断窗口中的时序
# 是否存在异常. 考虑到时序本身存在依赖关系, AE 和 VAE 在
# 实现前会添加 LSTM 层. 具体在预测上的运用可参考如下 papers:
# 1. LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection https://arxiv.org/pdf/1607.00148.pdf
# 2. Time-series Extreme Event Forecasting with Neural Networks at Uber

class _AE:
    pass

class _VAE:
    pass
