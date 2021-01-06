from Q1 import MyNeuralNetwork
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# read the datasets and parse them
train_df = pd.read_csv('mnist_train.csv')
test_df = pd.read_csv('mnist_test.csv')

dataset = train_df.to_numpy()
testset = test_df.to_numpy()

X_train = dataset[:, 1:]
X_test = testset[:, 1:]

# create two different train_data files 
# Relu and linear take datasets scaled by a factor of 1/255
X_train1 = copy.deepcopy(X_train)
X_test1 = copy.deepcopy(X_test)
X_train1 = X_train/255.0
X_test1 = X_test/255.0

X_train2 = copy.deepcopy(X_train)
X_test2 = copy.deepcopy(X_test)

y_train = dataset[:, 0]
y_test = testset[:, 0]


# use relu activation function
relu = MyNeuralNetwork(5, [784, 256, 128, 64, 10], 'relu', 0.1, 'normal', 500, 100)
relu.fit(X_train1, y_train, True, X_test1, y_test)
print("Training Accuracy for Relu:", relu.score(X_train1, y_train))
probs = relu.predict_proba(X_test1)
print("Testing Accuracy for Relu:", relu.score(X_test1, y_test))
preds = relu.predict(X_test1)
# relu.save_weights()

# use sigmoid activation function
sigmoid = MyNeuralNetwork(5, [784, 256, 128, 64, 10], 'sigmoid', 0.1, 'normal', 500, 100)
sigmoid.fit(X_train2, y_train, True, X_test2, y_test)
print("Training Accuracy for Sigmoid:", sigmoid.score(X_train2, y_train))
probs = sigmoid.predict_proba(X_test2)
print("Testing Accuracy for Sigmoid:", sigmoid.score(X_test2, y_test))
preds = sigmoid.predict(X_test2)
# sigmoid.save_weights()

# use linear activation function
linear = MyNeuralNetwork(5, [784, 256, 128, 64, 10], 'linear', 0.1, 'normal', 500, 100)
linear.fit(X_train1, y_train, True, X_test1, y_test)
print("Training Accuracy for Linear:", linear.score(X_train1, y_train))
probs = linear.predict_proba(X_test1)
print("Testing Accuracy for Linear:", linear.score(X_test1, y_test))
preds = linear.predict(X_test1)
# linear.save_weights()

# use tanh activation function
tanh = MyNeuralNetwork(5, [784, 256, 128, 64, 10], 'tanh', 0.1, 'normal', 500, 100)
tanh.fit(X_train2, y_train, True, X_test2, y_test)
print("Training Accuracy for Tanh:", tanh.score(X_train2, y_train))
probs = tanh.predict_proba(X_test2)
print("Testing Accuracy for Tanh:", tanh.score(X_test2, y_test))
preds = tanh.predict(X_test2)
# tanh.save_weights()


A_list, Z_list = relu.forward(X_test1)
A_last_hidden = A_list[-2]
A_last_hidden = A_last_hidden.T

# code for tsne visualisation
tsne=TSNE(n_components = 2)
emb=tsne.fit_transform(A_last_hidden)

sns.scatterplot(x = emb[:, 0], y = emb[:, 1], hue = y_test, legend = "full", palette = sns.color_palette("husl", n_colors = len(set(y_test))))
plt.show()


# sklearn implementation of MLP classifier
relu_sklearn = MLPClassifier(random_state = 10, activation='relu', max_iter=100, learning_rate_init=0.1, hidden_layer_sizes=(256,128,64), batch_size=500, solver = 'sgd').fit(X_train1, y_train)
print("Sklearn with Relu Testing Accuracy:", relu_sklearn.score(X_test1, y_test))


sigmoid_sklearn = MLPClassifier(random_state = 10, activation='logistic', max_iter=100, learning_rate_init=0.1, hidden_layer_sizes=(256,128,64), batch_size=500, solver = 'sgd').fit(X_train2, y_train)
print("Sklearn with Sigmoid Testing Accuracy:", sigmoid_sklearn.score(X_test2, y_test))

linear_sklearn = MLPClassifier(random_state = 10, activation='identity', max_iter=100, learning_rate_init=0.1, hidden_layer_sizes=(256,128,64), batch_size=500).fit(X_train1, y_train)
print("Sklearn with Linear Testing Accuracy:", linear_sklearn.score(X_test1, y_test))


tanh_sklearn = MLPClassifier(random_state = 10, activation='tanh', max_iter=100, learning_rate_init=0.1, hidden_layer_sizes=(256,128,64), batch_size=500, solver = 'sgd').fit(X_train2, y_train)
print("Sklearn with Tanh Testing Accuracy:", tanh_sklearn.score(X_test2, y_test))