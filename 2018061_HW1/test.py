from scratch import MyLinearRegression, MyLogisticRegression, MyPreProcessor, EDA
import numpy as np

preprocessor = MyPreProcessor()
linear = MyLinearRegression()
logistic = MyLogisticRegression()

X0, y0 = preprocessor.pre_process(0)
X1, y1 = preprocessor.pre_process(1)
X2, y2 = preprocessor.pre_process(2)


linear.CallKfoldCV(X0, y0, loss_type = 0, dataset = 0)
# Best fold comes out to be 7

linear.CallKfoldCV(X0, y0, loss_type = 1, dataset = 0)
# Best fold comes out to be 6

linear.CallKfoldCV(X1, y1, loss_type = 0, dataset = 1)
# Best fold comes out to be 10

linear.CallKfoldCV(X1, y1, loss_type = 1, dataset = 1)
# Best fold comes out to be 10

linear.KfoldCV(X0, y0, num_folds = 7, loss_type = 0, dataset = 0, flag = 0, plot = True)
# Best RMSE error: 2.7210663479424273, on split 4

linear.KfoldCV(X0, y0, num_folds = 6, loss_type = 1, dataset = 0, flag = 0, plot = True)
# Best MAE error: 1.7737218783162927, on split 4

linear.KfoldCV(X1, y1, num_folds = 10, loss_type = 0, dataset = 1, flag = 0, plot = True)
# Best RMSE error: 1.2249934336816115, on split 5

linear.KfoldCV(X1, y1, num_folds = 10, loss_type = 1, dataset = 1, flag = 0, plot = True)
# Best MAE error: 0.5931819212384858 on split 5

# Computing the training and validation loss
# for model based on dataset 0 and with RMSE loss function 
# on the best split which is on split 2 when k = 7
linear.Question1_partE(X0, y0, 0, 7)

# Exploratory Data Analysis on Dataset 2
EDA()

logistic.SplitRunAndPlot(X2, y2, type_GD = 0, learning_rate = 0.03, num_iterations = 10000)
# Running SGD using optimum parameters, learning_rate = 0.01, no. of iterations = 10000

logistic.SplitRunAndPlot(X2, y2, type_GD = 0, learning_rate = 0.0001, num_iterations = 100000)
# Running SGD using learning_rate = 0.0001, no. of iterations = 100000

logistic.SplitRunAndPlot(X2, y2, type_GD = 0, learning_rate = 0.01, num_iterations = 10000)
# Running SGD using learning_rate = 0.01, no. of iterations = 10000

logistic.SplitRunAndPlot(X2, y2, type_GD = 0, learning_rate = 10, num_iterations = 1000)
# Running SGD using learning_rate = 10, no. of iterations = 1000


logistic.SplitRunAndPlot(X2, y2, type_GD = 1, learning_rate = 0.03, num_iterations = 10000)
# Running BGD using optimum parameters, learning_rate = 0.01, no. of iterations = 10000

logistic.SplitRunAndPlot(X2, y2, type_GD = 1, learning_rate = 0.0001, num_iterations = 100000)
# Running BGD using learning_rate = 0.0001, no. of iterations = 100000

logistic.SplitRunAndPlot(X2, y2, type_GD = 1, learning_rate = 0.01, num_iterations = 10000)
# Running BGD using learning_rate = 0.01, no. of iterations = 10000

logistic.SplitRunAndPlot(X2, y2, type_GD = 1, learning_rate = 10, num_iterations = 1000)
# Running BGD using learning_rate = 10, no. of iterations = 1000






