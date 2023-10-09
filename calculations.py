import numpy as np


class Calculations:
    @staticmethod
    def entropy(y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    @staticmethod
    def gini_index(y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls ** 2
        return 1 - gini

    @staticmethod
    def information_gain(parent, l_child, r_child, mode="entropy"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "gini":
            gain = Calculations.gini_index(parent) - (
                        weight_l * Calculations.gini_index(l_child) + weight_r * Calculations.gini_index(r_child))
        else:
            gain = Calculations.entropy(parent) - (
                        weight_l * Calculations.entropy(l_child) + weight_r * Calculations.entropy(r_child))
        return gain
