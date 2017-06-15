import numpy as np
from decision_tree import *


class RandomForest(object):
    def __init__(self, depth_max=10, obs_min=10, tree_n=20, feature_n=10):
        self.depth_max = depth_max
        self.obs_min = obs_min
        self.tree_n = tree_n
        self.feature_n = feature_n
        self.trees = []

    def model_fitting(self, data, label):
        obs_count, feature_count = data.shape
        for n in range(self.tree_n):
            # bootstrapping
            random_rows = np.random.randint(0, obs_count, obs_count)
            # get random features
            random_features = np.random.choice(feature_count,
                                               self.feature_n,
                                               replace=False)
            random_set = data[random_rows, :][:, random_features]
            random_label = label[random_rows]
            dt = DecisionTree(self.depth_max, self.obs_min)
            dt.model_fitting(random_set, random_label)
            self.trees += [(random_features, dt)]

    def predict(self, test_data):
        all_votes = []
        for tree in self.trees:
            random_features, decision_tree = tree
            test_set = test_data[:, random_features]
            result = decision_tree.predict_prob(test_set)
            all_votes += [result]
        probability = np.mean(np.array(all_votes).T, axis=1)
        prediction = np.where(probability > 0.5, 0, 1)
        return prediction

