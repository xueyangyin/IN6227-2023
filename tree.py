import multiprocessing

import numpy as np

from calculations import Calculations
from node import Node


def calculate_leaf_value(label):
    label = list(label)
    return max(label, key=label.count)


def split(dataset, feature_index, threshold):
    dataset_left, dataset_right = [], []
    for row in dataset:
        if row[feature_index] <= threshold:
            dataset_left.append(row)
        else:
            dataset_right.append(row)
    return np.array(dataset_left), np.array(dataset_right)


def find_best_split(args):
    dataset, feature_idx = args
    best_split = {}
    highest_info_gain = -float("inf")

    feature_values = dataset[:, feature_idx]
    unique_thresholds = np.unique(feature_values)

    for threshold in unique_thresholds:
        dataset_left, dataset_right = split(dataset, feature_idx, threshold)

        if len(dataset_left) > 0 and len(dataset_right) > 0:
            y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
            current_info_gain = Calculations.information_gain(y, left_y, right_y, "gini")

            if current_info_gain > highest_info_gain:
                best_split = {
                    "feature_idx": feature_idx,
                    "threshold": threshold,
                    "dataset_left": dataset_left,
                    "dataset_right": dataset_right,
                    "info_gain": current_info_gain
                }
                highest_info_gain = current_info_gain

    return best_split


def get_best_split(dataset, num_features):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        args = [(dataset, i) for i in range(num_features)]
        results = pool.map(find_best_split, args)

    # Find the best split
    best_split = max(results, key=lambda x: x.get("info_gain", -float("inf")))

    return best_split


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=2):
        # root node
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def init(self, features, label):
        dataset = np.concatenate((features, label), axis=1)
        self.root = self.build(dataset)

    # Build the decision tree
    def build(self, dataset, curr_depth=0):
        features, label = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(features)

        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = get_best_split(dataset, num_features)
            if best_split["info_gain"] > 0:
                # Build the left
                left = self.build(best_split["dataset_left"], curr_depth + 1)
                # Build the right
                right = self.build(best_split["dataset_right"], curr_depth + 1)
                return Node(best_split["feature_idx"], best_split["threshold"],
                            left, right, best_split["info_gain"])

        leaf_value = calculate_leaf_value(label)
        return Node(value=leaf_value)

    # Do prediction
    def predict(self, features):
        predictions = [self._predict(feature, self.root) for feature in features]
        return predictions

    def _predict(self, feature, tree):
        if tree.value is not None:
            return tree.value
        feature_val = feature[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._predict(feature, tree.left)
        else:
            return self._predict(feature, tree.right)
