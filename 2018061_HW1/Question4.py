import pandas as pd
import numpy as np
from scratch import MyLogisticRegression

# reading dataset into pandas dataframe
df = pd.read_csv("Q4_Dataset.txt", sep = "   ", names = ["Class", "percentage", "age"])

# divding dataset into X and y and 
# adding columns of 1 to X for bias 
X = df[df.columns[1:]].values
y = df[[df.columns[0]]].values
num_rows, num_cols = X.shape
new_column = np.ones([num_rows, 1])
X = np.append(X, new_column, axis = 1)
y = y.reshape(-1);


# fitting data into my BGD implementaion
logistic = MyLogisticRegression()
logistic.fit(X, y, type_GD = 1, learning_rate = 0.03, num_iterations = 10000)

# gettting theta
parameters = logistic.parameters
print(parameters)
print("Parameters are in the order beta_1, beta_2, beta_0")

print("exp(beta_1):", np.exp(parameters[1][0]))
print("exp(beta_2):", np.exp(parameters[0][0]))

# predicting for
Xtest = np.array([[75, 2, 1]])
ypred = np.matmul(Xtest, parameters)

ypred = 1/(1+np.exp(-ypred))

print("Probability:", ypred[0][0])