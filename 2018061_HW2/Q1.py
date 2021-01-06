import h5py
import numpy as np
import pandas as pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


def ReadDatasets(dataset="A"):
   """
   Reads the dataset file of A
   and returns 2 separated numpy arrays:
   X and y
   """
   if dataset == "A":
      file_name = "part_A_train.h5"

   with h5py.File(file_name, "r") as f:

      X = np.array((f['X']))
      y = np.array((f['Y']))
      new_y = []

      for i in range(len(y)):
         for j in range(len(y[0])):
            if y[i][j] == 1:
               new_y.append(j)
               break

      y = np.array(new_y)


   return X, y


def StratifiedSampling(X, y):
   """

   Performs stratified sampling on the input dataset
   (given in the form of numpy arrays).

   Returns 4 numpy arrays after performing stratified sampling:
   X_train, X_test, y_train, y_test
   """
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=10, stratify=y)

   c_train = [0] * 10

   for i in range(len(X_train)):
      c_train[y_train[i]] += 1

   c_test = [0] * 10

   for i in range(len(X_test)):
         c_test[y_test[i]] += 1

   print("Training frequencies:")
   for freq in c_train:
      print(freq)

   print("\nTesting frequencies:")
   for freq in c_test:
      print(freq)

   print("\nTraining percentages:")
   for freq in c_train:
      print(freq / len(X_train))

   print("\nTesting percentages:")
   for freq in c_test:
      print(freq / len(X_test))

   ss =  StandardScaler()
   
   X_train = ss.fit_transform(X_train)
   X_test = ss.transform(X_test)

   return X_train, X_test, y_train, y_test

def PerformPCAThenCallTsne(X_train, X_test, y_train, y_test):
   """
   Applies PCA with number of components as 25
   on the given input dataset.
   After applying PCA, Calls function to perform
   TSNE on the transformed dataset.
   """

   print("\nApplying PCA")

   pca = PCA(n_components = 25, random_state = 10) 

   X_train = pca.fit_transform(X_train)
   X_test = pca.transform(X_test)

   PerformLogisticRegression(X_train, X_test, y_train, y_test)
   
   print("Creating TSNE Plot")
   PerformTSNE(X_train, y_train)

def PerformSVDThenCallTsne(X_train, X_test, y_train, y_test):
   """
   Applies SVD with number of components as 25
   on the given input dataset.
   After applying PCA, Calls function to perform
   TSNE on the transformed dataset.
   """

   
   print("\nApplying SVD")

   svd = TruncatedSVD(n_components = 25, random_state = 10)
   X_train = svd.fit_transform(X_train)
   X_test = svd.transform(X_test)


   PerformLogisticRegression(X_train, X_test, y_train, y_test)

   print("Creating TSNE Plot")
   PerformTSNE(X_train, y_train)



def PerformTSNE(X_train, y_train):
   """
   Applies TSNE on the given input 
   dataset and plots a 2 dimensional 
   space plot.
   """
   tsne = TSNE(n_components=2, random_state = 10)
   tsne_emb = tsne.fit_transform(X_train)
   sns.scatterplot(x = tsne_emb[:, 0], y = tsne_emb[:, 1], hue = y_train, legend = "full", palette = sns.color_palette("husl", n_colors = len(set(y_train))))
   plt.show()

def PerformLogisticRegression(X_train, X_test, y_train, y_test):
   """
   Fits a logistic regression model on the input 
   dataset and the predict the accuracy for the test set.
   """

   logistic = LogisticRegression(max_iter = 1000, random_state = 10)

   # fitting model on training data
   logistic.fit(X_train, y_train)

   # getting accuracy scores
   test_score = logistic.score(X_test, y_test)
   # train_score = logistic.score(X_train, y_train)

   print("Testing accuracy:", test_score)




X, y = ReadDatasets("A")
X_train, X_test, y_train, y_test = StratifiedSampling(X, y)
PerformPCAThenCallTsne(X_train, X_test, y_train, y_test)
PerformSVDThenCallTsne(X_train, X_test, y_train, y_test)
