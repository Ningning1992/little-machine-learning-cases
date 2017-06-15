import scipy.io
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

# Problem2
# load data
letters = scipy.io.loadmat('data/letters_data.mat')

# get the data, center and normalize the data
data = letters['train_x']/255
mean_vector = np.mean(data, axis=0)
for i in range(data.shape[1]):
    data[:, i] = data[:, i] - mean_vector[i]
labels = letters['train_y']
test = letters['test_x']/255
for i in range(test.shape[1]):
    test[:, i] = test[:, i] - mean_vector[i]

# combine the training example with its label for each training example
new = np.concatenate((data, labels), axis=1)

# shuffle the data and slice for the entire data set
np.random.seed(289)
np.random.shuffle(new)
new_label = new[:, -1]
new_set = new[:, :-1]
enc = OneHotEncoder(sparse=False)
new_label = new_label.reshape((len(new_label), 1))
new_label_enc = enc.fit_transform(new_label).T

row = new_set.shape[0]
ones = np.ones(row, dtype=float)
ones = ones.reshape((row, 1))
new_set = np.concatenate((new_set, ones), axis=1)

# save 20% training samples as a validation set
train_data = new[0:int(len(new)*0.8)]
validation_data = new[int(len(new)*0.8)::]

# slice the data to get labels
train_label = train_data[:, -1]
train_set = train_data[:, :-1]
validation_label = validation_data[:, -1]
validation_set = validation_data[:, :-1]

# preprocessing the data and labels
enc = OneHotEncoder(sparse=False)
train_label = train_label.reshape((len(train_label), 1))
train_label_enc = enc.fit_transform(train_label).T
validation_label = validation_label.reshape((len(validation_label), 1))
validation_label_enc = enc.fit_transform(validation_label).T

row1 = train_set.shape[0]
ones_1 = np.ones(row1, dtype=float)
ones_1 = ones_1.reshape((row1, 1))
train_set = np.concatenate((train_set, ones_1), axis=1)

row2 = validation_set.shape[0]
ones_2 = np.ones(row2, dtype=float)
ones_2 = ones_2.reshape((row2, 1))
validation_set = np.concatenate((validation_set, ones_2), axis=1)

row3 = test.shape[0]
ones_3 = np.ones(row3, dtype=float)
ones_3 = ones_3.reshape((row3, 1))
test = np.concatenate((test, ones_3), axis=1)


# train the neural network
nn = NeuralNetwork()
w, v = nn.train(train_set, train_label_enc, 3, 0.005, 0.01)
y_hat_train = nn.predict(train_set)
y_hat_val = nn.predict(validation_set)

accur_train = nn.accuracy(y_hat_train, train_label)
accur_val = nn.accuracy(y_hat_val, validation_label)

print(accur_train)
print(accur_val)


# plot the loss with number of iterations
# In order to make the training faster, I only calculated loss for this problem
# and made the loss calculation part as comments in my code. If you want to
# calculate the loss with my code, please make that part in neural_network.py
# as code instead of comments.
nn = NeuralNetwork()
w, v, loss = nn.train(train_set, train_label_enc, 1, 0.005, 0.01)
iteration = range(100, 99840, 100)

plt.figure(figsize=(10, 6))
plt.plot(iteration, loss)
plt.ylabel("loss")
plt.xlabel("number of iterations")
plt.title("loss plot")
plt.savefig('loss.png')

# Kaggle
nn = NeuralNetwork()
w1, v1 = nn.train(new_set, new_label_enc, 3, 0.005, 0.01)
y_hat = nn.predict(test)
np.savetxt("output_letters.csv", y_hat, delimiter=",")




# Problem3
nn = NeuralNetwork()
w3, v3 = nn.train(train_set, train_label_enc, 3, 0.005, 0.01)
y_hat_val = nn.predict(validation_set)
y_hat_val = y_hat_val.astype(int)

y_val = validation_label.reshape((1, len(validation_label)))
y_val = y_val[0]
y_val = y_val.astype(int)
letters = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z'])

index_correct = np.where(y_hat_val == y_val)
index_correct = index_correct[0]

index_incorrect = np.where(y_hat_val != y_val)
index_incorrect = index_incorrect[0]


def visual_correct(index_n):
    x = validation_set[index_correct[index_n], :-1]
    x = x.reshape(28, 28)
    y = y_val[index_correct[index_n]]

    plt.figure(figsize=(10, 6))
    plt.imshow(x)
    plt.title("True label: {} and Predicted label: {}".format(
        letters[y-1], letters[y_hat_val[index_n]-1]))
    plt.savefig('correct_x{}.png'.format(index_n+1))


def visual_incorrect(index_n):
    x = validation_set[index_incorrect[index_n], :-1]
    x = x.reshape(28, 28)
    y = y_val[index_incorrect[index_n]]

    plt.figure(figsize=(10, 6))
    plt.imshow(x)
    plt.title("True label: {} and Predicted label: {}".format(
        letters[y - 1], letters[y_hat_val[index_n] - 1]))
    plt.savefig('incorrect_x{}.png'.format(index_n + 1))


# visualize 5 correctly classified points
visual_correct(0)
visual_correct(1)
visual_correct(2)
visual_correct(3)
visual_correct(4)

# visualize 5 incorrectly classified points
visual_incorrect(0)
visual_incorrect(1)
visual_incorrect(2)
visual_incorrect(3)
visual_incorrect(4)

