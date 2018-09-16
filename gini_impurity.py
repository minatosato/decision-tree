#
# Copyright (c) 2018 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Set, Any


def gini_impurity(target: np.ndarray, classes: Set[Any]) -> float:
    ret = 1.0
    if len(target) == 0:
        return ret
    for _class in classes:
        ret -= (len(target[target == _class]) / len(target))**2
    return ret


def calc_gain(feature: np.ndarray, target: np.ndarray, threshold: float) -> float:
    classes = set(target)
    target_left = target[feature > threshold]
    target_right = target[feature <= threshold]
    criterion_before = gini_impurity(target, classes)
    criterion_left = gini_impurity(target_left, classes)
    criterion_right = gini_impurity(target_right, classes)
    criterion_after = criterion_left * len(target_left) / len(target) + criterion_right * len(target_right) / len(target)
    gain = criterion_before - criterion_after
    return gain


df = pd.read_csv('./iris.csv')
x = df.iloc[:, :4].values
y = df.iloc[:, 4].values

fig = plt.figure()

ax = None
for i, column in enumerate(df.columns[:-1]):
    kwargs: dict = {} if ax is None else {'sharey': ax}
    ax = fig.add_subplot('22{0}'.format(i+1), **kwargs)
    ax.title.set_text(column)
    ax.grid()

    gain = []
    feature_values = df[column].values
    thresholds = list(set(feature_values))
    thresholds.sort()
    for threshold in thresholds:
        gain.append(calc_gain(feature_values, y, threshold))

    ax.plot(thresholds, gain)
plt.show()


