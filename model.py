#
# Copyright (c) 2018 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from sklearn.datasets import fetch_mldata
from queue import Queue
from typing import Optional, Union, List, Set, Any, Tuple, cast
# import matplotlib.pyplot as plt

iris = fetch_mldata('iris')
x: np.ndarray = iris.data
y: np.ndarray = iris.target


class DecisionTreeNode(object):
    def __init__(self, features: np.ndarray, target: Union[np.ndarray, List[int]]) -> None:
        self.left: Optional[DecisionTreeNode] = None
        self.right: Optional[DecisionTreeNode] = None
        self.threshold: Optional[float] = None
        self.feature_index: Optional[int] = None
        self.gain = 0.0
        self.has_child = False

        self.features = features
        self.target = target

    def _split_node(self, threshold: float, feature_index: int, gain: float) -> None:
        self.threshold = threshold
        self.feature_index = feature_index
        self.gain = gain

        left_indices = self.features[:, feature_index] > threshold
        right_indices = self.features[:, feature_index] <= threshold

        # Todo 枝刈り
        if len(left_indices) < 5 or len(right_indices) < 5:
            pass
        else:
            self.left = DecisionTreeNode(self.features[left_indices], self.target[left_indices])
            self.right = DecisionTreeNode(self.features[right_indices], self.target[right_indices])

            self.has_child = True


class DecisionTree(object):
    def __init__(self) -> None:
        self.root: Optional[DecisionTreeNode] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        self.root = DecisionTreeNode(X, y)
        nodes: Queue = Queue()
        nodes.put(self.root)

        while nodes.qsize() > 0:
            current_node: DecisionTreeNode = nodes.get()
            threshold, feature_index, gain = self.calc_best_gain(current_node.features, current_node.target)
            if gain > 0:
                current_node._split_node(threshold, feature_index, gain)
                if current_node.has_child:
                    nodes.put(current_node.left)
                    nodes.put(current_node.right)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise Exception('This instance is not fitted yet. Call "fit" function before calling "predict".')
        ret = []
        for sample in X:
            current_node: DecisionTreeNode = self.root
            while current_node.has_child:
                if sample[current_node.feature_index] > current_node.threshold:
                    current_node = cast(DecisionTreeNode, current_node.left)
                else:
                    current_node = cast(DecisionTreeNode, current_node.right)
            classes, counts = np.unique(current_node.target, return_counts=True)
            ret.append(classes[counts.argmax()])
        return np.array(ret)

    def gini_coef(self, target: np.ndarray, classes: Union[List[int], Set[int]]) -> float:
        ret = 1.0
        if len(target) == 0:
            return ret
        for _class in classes:
            ret -= (len(target[target == _class]) / len(target))**2
        return ret

    def calc_gain(self, feature: np.ndarray, target: np.ndarray, threshold: float) -> float:
        classes = set(target)
        target_left = target[feature > threshold]
        target_right = target[feature <= threshold]
        gini_coef = self.gini_coef(target, classes)
        gini_coef_left = self.gini_coef(target_left, classes)
        gini_coef_right = self.gini_coef(target_right, classes)
        gini_impurity = gini_coef_left * len(target_left) / len(target) + gini_coef_right * len(target_right) / len(target)
        gain = gini_coef - gini_impurity
        return gain

    def calc_best_gain(self, features: np.ndarray, target: np.ndarray) -> Tuple[float, int, float]:
        best_feature_index = -1
        best_gain = 0.0
        best_threshold = 0.0
        for feature_index in range(features.shape[1]):
            feature = features[:, feature_index]
            thresholds = list(set(feature))
            thresholds.sort()
            for threshold in thresholds:
                gain = self.calc_gain(feature, target, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold
        return best_threshold, best_feature_index, best_gain
