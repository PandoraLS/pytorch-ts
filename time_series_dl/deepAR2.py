# -*- coding: utf-8 -*-
# @Time    : 2020/8/24 9:02 上午
# @Author  : sen

"""
参考来源：https://www.jianshu.com/p/4cb550302963
pip install gluonts
"""
from gluonts.model import deepar
from gluonts.dataset import common
from gluonts.dataset.util import to_pandas
from gluonts.model.predictor import Predictor
import pandas as pd
import matplotlib.pyplot as plt
csv_path = '/Users/seenli/Documents/workspace/code/pytorch_learn2/time_series_DL/Twitter_volume_AMZN.csv'
df = pd.read_csv(csv_path,header=0,sep=',')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index(['timestamp'],inplace=True)
data = common.ListDataset([{'start': df.index[0], 'target': df.value[:"2015-04-22 21:00:00"]}], freq='H')#这个数据格式是固定的
estimator = deepar.DeepAREstimator(freq='H', prediction_length=24)
predictor = estimator.train(training_data=data)
for train_entry, predict_result in zip(data, predictor.predict(data)):
    to_pandas(train_entry)[-60:].plot(linewidth=2)
    predict_result.plot(color='g', prediction_intervals=[50.0, 90.0])
plt.grid(which='both')
plt.show()
##输出预测结果
prediction = next(predictor.predict(data))
print(prediction.mean)
prediction.plot(output_file='graph.png')

