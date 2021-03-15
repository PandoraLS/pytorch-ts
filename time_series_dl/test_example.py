# -*- coding: utf-8 -*-
# @Time    : 2020/8/21 2:26 下午
# @Author  : sen

"""
github链接：https://stackoverflow.com/questions/58506775/catching-details-of-exception-in-python

pip install pytorchts   # 不过在DataLoader的时候存在问题，所以就不使用该方法了
"""

import matplotlib.pyplot as plt
import pandas as pd
import torch

from pts.dataset import ListDataset
from pts.model.deepar import DeepAREstimator
from pts import Trainer
from pts.dataset import to_pandas

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv"
path = "/Users/seenli/Documents/workspace/code/pytorch_learn2/time_series_DL/Twitter_volume_AMZN.csv"
# df = pd.read_csv(url, header=0, index_col=0, parse_dates=True)
df = pd.read_csv(path, header=0, index_col=0, parse_dates=True)


df[:100].plot(linewidth=2)
plt.grid(which='both')
plt.show()


training_data = ListDataset(
    [{"start": df.index[0], "target": df.value[:"2015-04-05 00:00:00"]}],
    freq = "5min"
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

estimator = DeepAREstimator(freq="5min",
                            prediction_length=12,
                            input_size=43,
                            trainer=Trainer(epochs=20,
                                            device=device))


# print('.....')

predictor = estimator.train(training_data=training_data)

test_data = ListDataset(
    [{"start": df.index[0], "target": df.value[:"2015-04-15 00:00:00"]}],
    freq = "5min",

)

for test_entry, forecast in zip(test_data, predictor.predict(test_data)):
    to_pandas(test_entry)[-60:].plot(linewidth=2)
    forecast.plot(color='g', prediction_intervals=[50.0, 90.0])
plt.grid(which='both')
plt.show()



