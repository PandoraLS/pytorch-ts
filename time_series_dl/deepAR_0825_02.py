# -*- coding: utf-8 -*-
# @Time    : 2020/8/25 3:13 下午
# @Author  : sen


"""
参考链接：
https://github.com/zmkwjx/GluonTS-Learning-in-Action/blob/master/chapter-2/document/MXNET%E4%B9%8BGluonTS%E5%AD%A6%E4%B9%A0%E6%89%8B%E5%86%8C%EF%BC%9A%E7%AC%AC%E4%BA%8C%E7%AB%A0%E3%80%8ADeepAR%E7%9A%84%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E3%80%8B.md
"""

from gluonts.model import deepar
from gluonts.dataset import common
from gluonts.dataset.util import to_pandas
from gluonts.model.predictor import Predictor
from gluonts.trainer import Trainer
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

train_data = common.FileDataset("此处填入训练数据文件夹的绝对路径", freq="H")
test_data  = common.FileDataset("此处填入需要预测数据文件夹的绝对路径", freq="H")

estimator = deepar.DeepAREstimator(
    prediction_length=24,
    context_length=100,
    use_feat_static_cat=True,
    use_feat_dynamic_real=True,
    num_parallel_samples=100,
    cardinality=[2,1],
    freq="H",
    trainer=Trainer(ctx="cpu", epochs=200, learning_rate=1e-3)
)
predictor = estimator.train(training_data=train_data)

for test_entry, forecast in zip(test_data, predictor.predict(test_data)):
    to_pandas(test_entry)[-100:].plot(figsize=(12, 5), linewidth=2)
    forecast.plot(color='g', prediction_intervals=[50.0, 90.0])
plt.grid(which='both')
plt.legend(["past observations", "median prediction", "90% prediction interval", "50% prediction interval"])
plt.show()

prediction = next(predictor.predict(test_data))
print(prediction.mean)
prediction.plot(output_file='graph.png')