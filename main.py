#
# Copyright (c) 2018 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import KFold
from model import DecisionTree

iris = fetch_mldata('iris')
x: np.ndarray = iris.data
y: np.ndarray = iris.target

kf = KFold(n_splits=5, shuffle=True)

for i, (train, test) in enumerate(kf.split(x)):
    clf = DecisionTree().fit(x[train], y[train])
    accuracy = (clf.predict(x[test]) == y[test]).sum() / len(y[test])
    print('{0}-validation accuracy: {1}'.format((i+1), accuracy))
