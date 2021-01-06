import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

np.random.seed(0) 

def ReadDataset():
	"""
   Reads the height-weight datset
   and returns 2 separated numpy arrays:
   X and y
   """

	df = pd.read_csv("weight-height.csv")

	df = df[df.columns[1:]]
	
	X = df["Height"].values
	y = df["Weight"].values


	X = X.reshape(-1, 1)

	return X, y

def CreateAndPredictOnBootstrapSample(X_train, X_test, y_train, y_test):
	"""
	Creates a random bootstrap sample by taking rows
	of X_train. The number of data elements(rows) in the 
	bootstrap sample is equal to the number of data elements(rows)
	in X_train

	Fits a linear regression model on the bootstrap sample
	and predicts values for X_test
	
	The function returns the mean of the predicted values.
	 """

	n = len(X_train)

	X_sampled = []
	y_sampled = []
	
	for i in range(n):
		random_row = np.random.randint(0, n)		
		X_sampled.append(X_train[random_row])
		y_sampled.append(y_train[random_row])


	reg = LinearRegression()
	reg.fit(X_sampled, y_sampled)

	y_pred = reg.predict(X_test)


	h_b = np.mean(y_pred)

	return h_b


X, y = ReadDataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

n_samples = 1000

h_b_list = []

for i in range(n_samples):
	h_b = CreateAndPredictOnBootstrapSample(X_train, X_test, y_train, y_test)
	h_b_list.append(h_b)

h_b_list = np.array(h_b_list)

h_bar = np.mean(h_b_list)

y_bar = np.mean(y_test)

bias = h_bar - y_bar
variance = np.var(h_b_list, ddof = 1)

print("Bias:", bias)
print("Variance:", variance)

h_b_minus_y = h_b_list-y_bar
h_b_minus_y = h_b_minus_y**2
mse = np.sum(h_b_minus_y)
mse = mse/(n_samples-1)

print("MSE:", mse)

result = mse - bias**2 - variance
print("MSE - Bias^2 - Variance:", result)

