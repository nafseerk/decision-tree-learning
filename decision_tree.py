from data_helper import DataHelper
from decision_tree_node import DecisionTreeNode
from util import get_best_attribute


class DecisionTree(object):

    def __init__(self, root_node):
        self.__root = root_node

    def get_root_node(self):
        return self.__root

    def print_tree(self):
        pass


class DecisionTreeLearning(object):

    @classmethod
    def learn(cls, data, attributes, default):

        if data.empty:
            return DecisionTree(default)
        elif is_date_same_class(data):
            node = DecisionTreeNode(data.iloc[0][DataHelper.get_target_class()], is_leaf_node=True)
            return DecisionTree(node)
        elif not attributes:
            return mode(data)
        else:
            best_attribute, threshold = get_best_attribute(data, attributes)[:-1]
            decision_tree_root = DecisionTreeNode(best_attribute, threshold)
            subtree_attributes = [attr for attr in attributes if attr != best_attribute]
            parent_data_mode = mode(data)
            left_subtree_data = data.loc[data[best_attribute] < threshold]
            right_subtree_data = data.loc[data[best_attribute] >= threshold]
            left_subtree = DecisionTreeLearning.learn(left_subtree_data, subtree_attributes, parent_data_mode)
            right_subtree = DecisionTreeLearning.learn(right_subtree_data, subtree_attributes, parent_data_mode)
            decision_tree_root.set_child(left_subtree.get_root_node(), 'left')
            decision_tree_root.set_child(right_subtree.get_root_node(), 'right')
            return DecisionTree(decision_tree_root)

    @classmethod
    def predict(cls, decision_tree, data, report_accuracy=False):
        if data.empty or decision_tree is None:
            raise ValueError('Bad arguments')

        if report_accuracy and DataHelper.get_target_class() not in data:
            raise ValueError('Cannot report accuracy. Ground truth not available')

        predictions = []
        correctly_classified = 0
        total = data.shape[0]

        for index, row in data.iterrows():
            current_node_test = decision_tree.get_root_node()
            while not current_node_test.is_leaf_node():
                if row[current_node_test.get_attribute()] < current_node_test.get_threshold():
                    current_node_test = current_node_test.get_child('left')
                else:
                    current_node_test = current_node_test.get_child('right')
            prediction = current_node_test.get_attribute()
            if prediction == row[DataHelper.get_target_class()]:
                correctly_classified += 1
            predictions.append(prediction)

        if report_accuracy:
            accuracy = correctly_classified / total * 100
            print('Accuracy = %.2f %%' % accuracy)

        return predictions


def mode(data):
    if data is None:
        raise ValueError('Data is empty')

    if DataHelper.get_target_class() not in data:
        raise ValueError('Data does not contain target class information')

    positive_count = data.loc[data[DataHelper.get_target_class()] == 'colic'].shape[0]
    negative_count = data.loc[data[DataHelper.get_target_class()] == 'healthy'].shape[0]

    if positive_count > negative_count:
        return DecisionTreeNode('colic', is_leaf_node=True)
    else:
        return DecisionTreeNode('healthy', is_leaf_node=True)


def is_date_same_class(data):
    if data is None:
        raise ValueError('Data is empty')

    if DataHelper.get_target_class() not in data:
        raise ValueError('Data does not contain target class information')

    positive_count = data.loc[data[DataHelper.get_target_class()] == 'colic'].shape[0]
    total_count = data.shape[0]

    return positive_count == total_count or positive_count == 0


if __name__ == '__main__':
    trainData = DataHelper.get_train_data()
    testData = DataHelper.get_test_data()

    learned_decision_tree = DecisionTreeLearning.learn(trainData, DataHelper.get_attributes(), mode(trainData))
    print('==On Test Data==')
    predictedY = DecisionTreeLearning.predict(learned_decision_tree, testData, report_accuracy=True)
    print('Predictions:')
    for i in range(len(predictedY)):
        print('%d. %s' % (i + 1, predictedY[i]))
    print()

    print('==On Training Data==')
    training_predictions = DecisionTreeLearning.predict(learned_decision_tree, trainData, report_accuracy=True)
    print('Predictions:')
    for i in range(len(training_predictions)):
        print('%d. %s' % (i + 1, training_predictions[i]))

