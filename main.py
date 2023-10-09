import time

import pandas as pd

from tree import DecisionTree

if __name__ == '__main__':
    col_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income"
    ]
    start_time = time.time()

    train_data = pd.read_csv("data\\adult.data", skiprows=1, header=None, names=col_names)
    test_data = pd.read_csv("data\\adult.test", skiprows=1, header=None, names=col_names)

    # Handle missing value by replace them to 0
    # train_data = train_data.replace(['?', ' ?'], '0')
    # test_data = test_data.replace(['?', ' ?'], '0')

    features_train = train_data.iloc[:, :-1].values
    label_train = train_data.iloc[:, -1].values.reshape(-1, 1)

    features_test = test_data.iloc[:, :-1].values
    label_test = test_data.iloc[:, -1].values.reshape(-1, 1)

    decisionTree = DecisionTree(min_samples_split=3, max_depth=3)
    decisionTree.init(features_train, label_train)

    Y_pred = decisionTree.predict(features_test)
    correct = 0

    for index, value in enumerate(Y_pred):
        cleaned_pred_value = value.replace(" ", "").replace(".", "")
        cleaned_real_value = label_test[index][0].replace(" ", "").replace(".", "")
        if cleaned_pred_value == cleaned_real_value:
            correct += 1

    print(f"Accuracy: {correct / len(Y_pred) * 100}%")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time: {elapsed_time} seconds")
