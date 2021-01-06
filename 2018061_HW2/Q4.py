import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.naive_bayes import GaussianNB
import copy


def ReadDatasets(dataset="A"):
   """
   Reads the dataset file of the given input dataset
   and returns 2 separated numpy arrays:
   X and y
   """ 
   if dataset == "A":
      file_name = "part_A_train.h5"
   else:
      file_name = "part_B_train.h5"


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

class MyGNB():
    """
    Class which contains the implementation of 
    Gaussian Naive Bayes classifier.
    """

    def __init__(self):
        """
        contains instance variables which are created during 
        model fitting
        """
        self.mean_list = []
        self.sd_list = []
        self.p_y_list = []

    def fit(self, X, y):
        """
        Fits the model on the given input numpy arrays
        Also creates the instance variables:
        mean_list, sd_list, indexes and p_y_list.
        """
        n = len(set(y))
    
        mean_list = []
        sd_list = []
        indexes = []
        p_y_list = [] 
        
        for i in range(n):
            indexes = np.where(y == i)[0]
            X_i = X[indexes]
            means = np.mean(X_i, axis = 0)
            mean_list.append(means)
            
            sds = np.std(X_i, axis = 0)
            sds = sds + 10**(-2)
            
            sd_list.append(sds)
            p_y_list.append(len(indexes)/X.shape[0])
            
            
        self.mean_list = np.array(mean_list)
        self.sd_list = np.array(sd_list)
        self.p_y_list = np.array(p_y_list)
        
        
    def predict(self, X):
        """
        Predicts the values of the class variables
        on the data in the given numpy array
        """
        
        y_pred = []
        
        n = len(self.mean_list)
        
        mean_matrix = self.mean_list 
        sd_matrix = self.sd_list 
        
        probs = None

        for i in range(n):
            X_i = copy.deepcopy(X)
            X_i = X_i - mean_matrix[i]
            
            X_i = X_i**2
            X_i = X_i/(2*sd_matrix[i]**2)
            X_i = np.exp(-X_i)
            X_i = X_i/(2*np.pi*sd_matrix[i]**2)**0.5
            X_i = np.log(X_i)
            X_i = np.sum(X_i, axis = 1)
            X_i = X_i + np.log(self.p_y_list[i])
            if probs is None:
                probs = X_i
            else:
                probs = np.column_stack((probs, X_i))
                
        for i in range(X.shape[0]):
             
            max_ind = 0
            maxx = probs[i][0]
            for j in range(n):
                if probs[i][j]>maxx:
                    maxx = probs[i][j]
                    max_ind = j
            y_pred.append(max_ind)
        
        return y_pred

        
    
    def score(self, y_test, y_pred):
        """
        Returns accuracy score on the input
        predicted and true values.
        """
        c = 0
        for i in range(len(y_test)):
            if y_test[i]==y_pred[i]:
                c += 1
        return c/len(y_test)
    
            
X, y = ReadDatasets("A")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = MyGNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("My implementation score on dataset A:", clf.score(y_test, y_pred))

clf = GaussianNB()
clf.fit(X_train, y_train)

test_score = clf.score(X_test, y_test)
print("Sklearn score on dataset A:", test_score)

X, y = ReadDatasets("B")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = MyGNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("My implementation score on dataset B:", clf.score(y_test, y_pred))

clf = GaussianNB()
clf.fit(X_train, y_train)

test_score = clf.score(X_test, y_test)
print("Sklearn score on dataset B:", test_score)

