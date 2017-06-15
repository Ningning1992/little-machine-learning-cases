import numpy as np
from collections import namedtuple
from operator import itemgetter


class Node(object):
    def __init__(self, data, label, type=None, split_variable=None, split_value=None):
        self.data = data
        self.label = label
        self.type = type
        self.split_variable = split_variable
        self.split_value = split_value
        self.left = None
        self.right = None

    # for each set of the node, calculate the probability of 0
    def prob_zero(self):
        count_0 = np.count_nonzero(self.label)
        n = len(self.label)
        prob_0 = (n-count_0)/n
        return prob_0

    def traversing(self, point):
        if point[int(self.split_variable)] <= self.split_value:
            # print(self.split_variable, "<=", self.split_value)
            if self.left.type == 'leaf':
                return self.prob_zero()
            return self.left.traversing(point)
        else:
            # print(self.split_variable, ">", self.split_value)
            if self.right.type == 'leaf':
                return self.prob_zero()
            return self.right.traversing(point)


class DecisionTree(object):
    def __init__(self, depth_max=5, obs_min=100):
        self.depth_max = depth_max
        self.obs_min = obs_min
        self.node_count = 1

    def entropy(self, prob_dic):
        # prob_dic (a dictionary contains left and right probability):
        # {'0': prob(num, den), '1': prob(num, den)}
        p0, p1 = prob_dic.values()
        if p0.den == 0:
            return -np.log2(1)
        elif p1.den == 0:
            return -np.log2(1)
        else:
            prob_0 = p0.num/p0.den
            prob_1 = 1-prob_0
            return -prob_0*np.log2(prob_0)-prob_1*np.log2(prob_1)

    def info_gain(self, entropy_parent, child_prob):
        # entropy_parent: the number calculated by entropy method
        # child_prob:tuple (prob_dic, prob_dic)
        left_child, right_child = child_prob
        left_den = left_child['0'].den
        right_den = right_child['0'].den
        total_den = left_den + right_den
        left_weight = left_den / total_den * self.entropy(left_child)
        right_weight = right_den / total_den * self.entropy(right_child)
        gain = entropy_parent - (left_weight + right_weight) / 2
        return gain

    def feature_split(self, label, feature_data, feature_name, entropy_parent):
        # Find best value to split for each feature
        # output:[('feature1', info_gain), ('feature2', info_gain), ...]
        prob = namedtuple('prob', ['num', 'den'])
        unique = np.unique(feature_data)

        feature_info_gain = []
        for value in unique:
            l_label = label[feature_data <= value]
            l1_n = np.count_nonzero(l_label)
            l0_n = len(l_label) - l1_n

            r_label = label[feature_data > value]
            r1_n = np.count_nonzero(r_label)
            r0_n = len(r_label) - r1_n

            l_total = l1_n + l0_n
            r_total = r1_n + r0_n

            # if observation in the children set is too small, stop split
            if min(l_total, r_total) <= self.obs_min:
                feature_info_gain += [(None, 0)]

            split_rule = "{} <= {}".format(feature_name, value)
            l_p = {'0': prob(l0_n, l_total), '1': prob(l1_n, l_total)}
            r_p = {'0': prob(r0_n, r_total), '1': prob(r1_n, r_total)}

            child_prob = (l_p, r_p)
            gain = self.info_gain(entropy_parent, child_prob)
            feature_info_gain += [(split_rule, gain)]

        return feature_info_gain

    def optimal_split(self, data, label):
        # find the best split rule
        prob = namedtuple('prob', ['num', 'den'])
        n_1 = np.count_nonzero(label)
        total = len(label)
        parent_prob = {
            '0': prob((total - n_1), total),
            '1': prob(n_1, total)
        }
        entropy_parent = self.entropy(parent_prob)

        feature_gain = []
        for feature_name, feature_data in enumerate(data.T):
            feature_gain += self.feature_split(label,
                                               feature_data,
                                               feature_name,
                                               entropy_parent)

        max_gain_rule = max(feature_gain, key=itemgetter(1))
        return max_gain_rule

    def get_child(self, node):
        # use best feature and value to split on in entire data set
        # returns: (Node(), Node())
        data = node.data
        label = node.label

        split_rule, info_gain = self.optimal_split(data, label)
        split_feature = split_rule.split()[0]
        split_value = float(split_rule.split()[2])

        index = data[:, split_feature] <= split_value
        l_set = data[index]
        l_label = label[index]
        r_set = data[~index]
        r_label = label[~index]

        left_child = Node(l_set,  l_label, split_variable=split_feature,
                          split_value=split_value)

        right_child = Node(r_set,  r_label, split_variable=split_feature,
                           split_value=split_value)

        return left_child, right_child

    def get_tree(self, data, label, depth=0):
        unique = np.unique(label)
        if len(unique) == 1:
            if unique == 0:
                return Node(data=None, label=0, type='leaf')
            elif unique == 1:
                return Node(data=None, label=1, type='leaf')

        elif data.shape[0] <= self.obs_min:
            return Node(data=None, label=None, type='leaf')

        elif depth == self.depth_max:
            return Node(data=None, label=None, type='leaf')

        else:
            tree = Node(data, label, type='internal')
            left, right = self.get_child(tree)
            tree.split_variable = left.split_variable
            tree.split_value = left.split_value

            tree.left = self.get_tree(left.data, left.label, depth + 1)

            tree.right = self.get_tree(right.data, right.label, depth + 1)
            return tree

    def model_fitting(self, data, label):
        self.mytree = self.get_tree(data, label)

    def predict_prob(self, test_data):
        prob = np.array([self.mytree.traversing(row) for row in test_data])
        return prob

    def predict(self, test_data):
        prob = self.predict_prob(test_data)
        result = np.where(prob > 0.5, 0, 1)
        return result
