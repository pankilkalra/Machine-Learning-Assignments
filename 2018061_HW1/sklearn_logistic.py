from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np

from scratch import MyPreProcessor

def RunLogisticRegression(Xtrain, ytrain, Xtest, ytest):
	# creating object for logistiv regretting
	logistic = LogisticRegression()

	# fitting model on training data
	logistic.fit(Xtrain, ytrain)

	# getting accuracy scores
	test_score = logistic.score(Xtest, ytest)
	train_score = logistic.score(Xtrain, ytrain)

	return train_score, test_score

def RunSGDClasifier(Xtrain, ytrain, Xtest, ytest):
	# creating model using alpha = 0.03, max_iter = 10000 
	model = SGDClassifier(alpha = 0.03, max_iter = 10000)
	
	# fitting model on training data
	model.fit(Xtrain, ytrain)
	
	# getting accuracy scores
	test_score = model.score(Xtest, ytest)
	train_score = model.score(Xtrain, ytrain)

	return train_score, test_score


# preprocessing dataset and divding it into
# training:validation:testing set in the ration 7:1:@
preprocessor = MyPreProcessor()

X, y = preprocessor.pre_process(2)

X_folds = np.array_split(X, 10)
y_folds = np.array_split(y, 10)


X_training_folds = []
y_training_folds = []

for i in range(7):
   X_training_folds.append(X_folds[i])
   y_training_folds.append(y_folds[i])

Xtrain = np.concatenate(X_training_folds)
ytrain = np.concatenate(y_training_folds)

X_testing_folds = []
y_testing_folds = []

for i in range(8, 10):
   X_testing_folds.append(X_folds[i])
   y_testing_folds.append(y_folds[i])

Xtest = np.concatenate(X_testing_folds)
ytest = np.concatenate(y_testing_folds)

# running logistic regression
train_score, test_score = RunLogisticRegression(Xtrain, ytrain, Xtest, ytest)
print("Logistic Regression Training Accuracy: " + str(train_score*100) + " Logistic Regression Test Accuracy: " + str(test_score*100))

# running SGD classifier
train_score, test_score = RunSGDClasifier(Xtrain, ytrain, Xtest, ytest)
print("SGDClassifier Training Accuracy: " + str(train_score*100) + " SGDClassifier Test Accuracy: " + str(test_score*100))

