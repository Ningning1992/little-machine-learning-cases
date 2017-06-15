import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
from decision_tree import *
from random_forest import *

train = pd.read_csv('census/train_data.csv')
label = train.label
train_set = train.drop('label', 1)
test_set = pd.read_csv('census/test_data.csv')


# data processing
# do vectorizing for categorical variable
# consider missing value as a different category when vectorizing
dv = DictVectorizer(sparse=False)
train_dict = train_set.T.to_dict().values()
test_dict = test_set.T.to_dict().values()
train_transformed = dv.fit_transform(train_dict)
test_transformed = dv.transform(test_dict)

# save 20% training samples as a validation set
train_data = train_transformed[0:int(len(train_transformed)*0.8)]
train_label = label[0:int(len(label)*0.8)]

validation_data = train_transformed[int(len(train_transformed)*0.8)::]
validation_label = label[int(len(label)*0.8)::]

# implement decision trees
census_tree = DecisionTree(depth_max=10, obs_min=20)
census_tree.model_fitting(train_data, train_label)

predict_train1 = census_tree.predict(train_data)
train_accuracy1 = sum(predict_train1 == train_label)/len(train_label)

predict_validation1 = census_tree.predict(validation_data)
validation_accuracy1 = sum(predict_validation1 == validation_label)\
                       /len(validation_label)

print("training accuracy for census--decision tree\n", train_accuracy1)
print("validation accuracy for census--decision tree\n", validation_accuracy1)


# implement random forest
census_rf = RandomForest(depth_max=10, obs_min=30, tree_n=10, feature_n=50)
census_rf.model_fitting(train_data, train_label)

predict_train2 = census_rf.predict(train_data)
train_accuracy2 = sum(predict_train2 == train_label)/len(train_label)

predict_validation2 = census_rf.predict(validation_data)
validation_accuracy2 = sum(predict_validation2 == validation_label)\
                       /len(validation_label)

print("training accuracy for census--random forest\n", train_accuracy2)
print("validation accuracy for census--random forest\n", validation_accuracy2)


# Kaggle submission
dt = DecisionTree(depth_max=12, obs_min=15)
dt.model_fitting(train_transformed, label)
result = dt.predict(test_transformed)
np.savetxt("output_census.csv", result, delimiter=",")


# report split rule for decision tree
two_point = validation_data[3:5, :]
two_point_label = validation_label[3:5]

dt = DecisionTree(depth_max=10, obs_min=20)
dt.model_fitting(train_data, train_label)
predict_label = dt.predict(two_point)

print(two_point_label)
print(predict_label)
print(dv.get_feature_names())


# report common splits made at the root node for random forest
rf_s = RandomForest(depth_max=3, obs_min=20, tree_n=10, feature_n=50)
rf_s.model_fitting(train_data, train_label)
predict_rf = rf_s.predict(two_point[:1, :])


# validation accuracy graph
validation_list = []
for i in range(1, 11):
    census_tree = DecisionTree(depth_max=i, obs_min=40)
    census_tree.model_fitting(train_data[0:6000], train_label[0:6000])

    prediction = census_tree.predict(validation_data[0:1500])
    accuracy = sum(prediction == validation_label[0:1500]) \
                           / len(validation_label)

    validation_list += [accuracy]

x = range(1, 11)
y = validation_list

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.ylabel("validation accuracy")
plt.xlabel("maximum depth")
plt.title("validation accuracy for decision tree")
plt.savefig('accuracy_plot.png')
