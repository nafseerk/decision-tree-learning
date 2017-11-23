class DecisionTreeNode(object):

    def __init__(self, attribute, threshold=None, is_leaf_node=False):
        self._attribute = attribute
        self._is_leaf_node = is_leaf_node
        if not is_leaf_node:
            self._threshold = threshold
            self._left_child = None
            self._right_child = None

    def get_attribute(self):
        return self._attribute

    def set_attribute(self, attribute):
        self._attribute = attribute

    def get_threshold(self):
        if not self._is_leaf_node:
            return self._threshold
        else:
            return None

    def set_threshold(self, threshold):
        if not self._is_leaf_node:
            self._threshold = threshold

    def get_child(self, which):
        if not self._is_leaf_node:
            if which == 'left':
                return self._left_child
            elif which == 'right':
                return self._right_child

        return None

    def set_child(self, child_node, which):
        if not self._is_leaf_node:
            if which == 'left':
                self._left_child = child_node
            elif which == 'right':
                self._right_child = child_node

    def is_leaf_node(self):
        return self._is_leaf_node

    def print_details(self):
        if self._is_leaf_node:
            print(self._attribute)
            return

        print('====Test %s <= %f====' % (self._attribute, self._threshold))
        print('Left Child: Test %s' % self._left_child.get_attribute())
        print('Right Child: Test %s' % self._right_child.get_attribute())


if __name__ == '__main__':
    decisionTreeNode = DecisionTreeNode('Height', 175.5)
    child1 = DecisionTreeNode('Weight', 1.54)
    child2 = DecisionTreeNode('HeartRate', 10.223)
    decisionTreeNode.set_child(child1, 'left')
    decisionTreeNode.set_child(child2, 'right')
    decisionTreeNode.print_details()
