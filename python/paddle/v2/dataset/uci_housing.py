# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
from common import download

__all__ = ['train', 'test']

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
MD5 = 'd4accdce7a25600298819f8e28e8d593'
feature_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT'
]

UCI_TRAIN_DATA = None
UCI_TEST_DATA = None


def feature_range(maximums, minimums):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    feature_num = len(maximums)
    ax.bar(range(feature_num), maximums - minimums, color='r', align='center')
    ax.set_title('feature scale')
    plt.xticks(range(feature_num), feature_names)
    plt.xlim([-1, feature_num])
    fig.set_figheight(6)
    fig.set_figwidth(10)
    if not os.path.exists('./image'):
        os.makedirs('./image')
    fig.savefig('image/ranges.png', dpi=48)
    plt.close(fig)


def load_data(filename, feature_num=14, ratio=0.8):
    global UCI_TRAIN_DATA, UCI_TEST_DATA
    if UCI_TRAIN_DATA is not None and UCI_TEST_DATA is not None:
        return

    data = np.fromfile(filename, sep=' ')
    data = data.reshape(data.shape[0] / feature_num, feature_num)
    maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(
        axis=0) / data.shape[0]
    feature_range(maximums[:-1], minimums[:-1])
    for i in xrange(feature_num - 1):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    offset = int(data.shape[0] * ratio)
    UCI_TRAIN_DATA = data[:offset]
    UCI_TEST_DATA = data[offset:]


def train():
    global UCI_TRAIN_DATA
    load_data(download(URL, 'uci_housing', MD5))

    def reader():
        for d in UCI_TRAIN_DATA:
            yield d[:-1], d[-1:]

    return reader


def test():
    global UCI_TEST_DATA
    load_data(download(URL, 'uci_housing', MD5))

    def reader():
        for d in UCI_TEST_DATA:
            yield d[:-1], d[-1:]

    return reader
