import math
from data_helper import DataHelper

'''
Records with isColic=colic is considered positive
and with isColic=healthy is considered negative
'''
TARGET_ATTRIBUTE = 'isColic'


def entropy(v1, v2):
    if v1 < 0 or v2 < 0:
        raise ValueError("Entropy is defined only for positive numbers")
    if v1 == 0 and v2 == 0:
        raise ValueError("Entropy not defined for both 0s")

    v1 = float(v1)
    v2 = float(v2)
    total = v1 + v2
    v1_ratio = v1/total
    v2_ratio = v2/total
    v1_term = 0.0 if (v1_ratio == 0.0) else - v1_ratio * math.log(v1_ratio, 2)
    v2_term = 0.0 if (v2_ratio == 0.0) else - v2_ratio * math.log(v2_ratio, 2)
    return v1_term + v2_term


def remainder(data, attribute):
    if data is None or attribute is None or attribute not in data:
        raise ValueError("Bad arguments")

    total = float(data.shape[0])
    unique_values = data[attribute].unique()
    result = 0.0

    for value in unique_values:
        filtered_data = data.loc[data[attribute] == value]
        total_with_this_value = float(filtered_data.shape[0])
        p_with_this_value = float(filtered_data.loc[data[TARGET_ATTRIBUTE] == 'colic'].shape[0])
        p_ratio = p_with_this_value/total_with_this_value
        n_with_this_value = float(filtered_data.loc[data[TARGET_ATTRIBUTE] == 'healthy'].shape[0])
        n_ratio = n_with_this_value/total_with_this_value
        result += total_with_this_value/total * entropy(p_ratio, n_ratio)

    return result


def information_gain(data, attribute):
    if data is None or attribute is None or attribute not in data:
        raise ValueError("Bad arguments")
    positive_count = float(data.loc[data[TARGET_ATTRIBUTE] == 'colic'].shape[0])
    negative_count = float(data.loc[data[TARGET_ATTRIBUTE] == 'healthy'].shape[0])
    total = positive_count + negative_count
    p_ratio = positive_count/total
    n_ratio = negative_count/total

    print('entropy=%f' % entropy(p_ratio, n_ratio))
    return entropy(p_ratio, n_ratio) - remainder(data, attribute)


if __name__ == '__main__':
    print(entropy(1/2, 1/2))
    trainX = DataHelper.get_train_x()
    trainY = DataHelper.get_train_y()
    trainData = DataHelper.get_train_data()
    testX = DataHelper.get_test_x()
    testY = DataHelper.get_test_y()
    testData = DataHelper.get_test_data()
    print(remainder(trainData, 'K'))
    print(information_gain(trainData, 'K'))
