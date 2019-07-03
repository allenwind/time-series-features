# 联合 backend 实现的特征提取模型，方便整合到 NN 的梯度流中.
# 特征上无法和 features.py 模块完全一致.

# 在 keras 中使用 Lambda 进行无状态计算.
# 通过在神经网络隐层中加入人工特征, 可以显著减少
# 隐层的宽度, 提高训练效率.
# 参考文档:
# https://www.tensorflow.org/api_docs/python/tf/py_function
