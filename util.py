import math
from data_helper import DataHelper

'''
Records with isColic=colic is considered positive
and with isColic=healthy is considered negative
'''
TARGET_ATTRIBUTE = 'isColic'
ATTRIBUTES = ['K',
              'Na',
              'CL',
              'HCO3',
              'Endotoxin',
              'Aniongap',
              'PLA2',
              'SDH',
              'GLDH',
              'TPP',
              'Breath rate',
              'PCV',
              'Pulse rate',
              'Fibrinogen',
              'Dimer',
              'FibPerDim'
             ]


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

    lowest_remainder = None
    best_threshold = None

    total = float(data.shape[0])
    sorted_unique_values = sorted(set(data[attribute]))

    for threshold in sorted_unique_values:
        remainder_value = 0.0
        filtered_data = data.loc[data[attribute] <= threshold]
        total_below_threshold = float(filtered_data.shape[0])
        if total_below_threshold > 0:
            p_below_threshold = float(filtered_data.loc[data[TARGET_ATTRIBUTE] == 'colic'].shape[0])
            p_ratio = p_below_threshold/total_below_threshold
            n_below_threshold = float(filtered_data.loc[data[TARGET_ATTRIBUTE] == 'healthy'].shape[0])
            n_ratio = n_below_threshold/total_below_threshold
            remainder_value += total_below_threshold/total * entropy(p_ratio, n_ratio)

        filtered_data = data.loc[data[attribute] > threshold]
        total_above_threshold = float(filtered_data.shape[0])
        if total_above_threshold > 0:
            p_above_threshold = float(filtered_data.loc[data[TARGET_ATTRIBUTE] == 'colic'].shape[0])
            p_ratio = p_above_threshold / total_above_threshold
            n_above_threshold = float(filtered_data.loc[data[TARGET_ATTRIBUTE] == 'healthy'].shape[0])
            n_ratio = n_above_threshold / total_above_threshold
            remainder_value += total_above_threshold / total * entropy(p_ratio, n_ratio)

        if lowest_remainder is None or remainder_value < lowest_remainder:
            lowest_remainder = remainder_value
            best_threshold = threshold

    return lowest_remainder, best_threshold


def information_gain(data, attribute):
    if data is None or attribute is None or attribute not in data:
        raise ValueError("Bad arguments")
    positive_count = float(data.loc[data[TARGET_ATTRIBUTE] == 'colic'].shape[0])
    negative_count = float(data.loc[data[TARGET_ATTRIBUTE] == 'healthy'].shape[0])
    total = positive_count + negative_count
    p_ratio = positive_count/total
    n_ratio = negative_count/total

    lowest_remainder, best_threshold = remainder(data, attribute)
    return entropy(p_ratio, n_ratio) - lowest_remainder, best_threshold


def get_best_attribute(data):
    max_information_gain = - 1
    best_attribute = None
    best_threshold = None

    for attribute in ATTRIBUTES:
        ig, threshold = information_gain(data, attribute)
        if ig > max_information_gain:
            max_information_gain = ig
            best_attribute = attribute
            best_threshold = threshold

    return best_attribute, best_threshold, max_information_gain


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
    print(get_best_attribute(trainData))
