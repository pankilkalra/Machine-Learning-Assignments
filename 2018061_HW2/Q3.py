import h5py
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.naive_bayes import GaussianNB
from itertools import product 
import matplotlib.pyplot as plt
import pickle

random_seed = 5

def ReadDatasets(dataset="A"):
	# Reads the given dataset and
	# returns separated X and y numpy
	# arrays
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

def DecisionTreeClassification(X_train, X_test, y_train, y_test, params, save_model = False, dataset = "A"):
	"""
	Fits a decision tree classifier on the given input numpy
	arrays. If parameters are provided for the model,
	fitting is done using them.
	
	The function returns the training and testing score after 
	computing the score for the testing data,

	The function also saves the model if the input save_model parameter
	is True.
	"""

	if params is None:
		clf = DecisionTreeClassifier(random_state = random_seed)
	else:
		clf = DecisionTreeClassifier(**params)


	clf.fit(X_train, y_train)

	val_score = clf.score(X_test, y_test)
	train_score = clf.score(X_train, y_train)

	if save_model == True:
		f = open("dt_model_" + dataset + ".pkl", "wb")
		pickle.dump(clf, f)
		f.close()


	return train_score, val_score

def GaussianNaiveBayesClassification(X_train, X_test, y_train, y_test, save_model = False, dataset = "A"):
	"""
	Fits a decision Gaussian Naive Bayes Classifier on the given input numpy
	arrays. 
	
	The function returns the training and testing score after 
	computing the score for the testing data,

	The function also saves the model if the input save_model parameter
	is True.
	"""


	clf = GaussianNB()
	clf.fit(X_train, y_train)

	val_score = clf.score(X_test, y_test)
	train_score = clf.score(X_train, y_train)

	if save_model == True:
		f = open("gnb_model_" + dataset + ".pkl", "wb")
		pickle.dump(clf, f)
		f.close()


	return train_score, val_score

def KfoldCV(X, y, num_folds = 4, classifier_type = 0, params = None, save_best_model = False, dataset = "A"):
	"""
	The function performs K fold Cross Validation on the given input numpy arrays.
	The number of folds are taken as input.

	If classifier_type is 0, then DT is used
	else GNB isused.

	Parameters for the model can be provided if the classifier is DT.

	Also, the best model would be saved if the save_best_model flag 
	is set to True.
	"""
	num_rows, num_cols = X.shape

	X_folds = np.array_split(X, num_folds)
	y_folds = np.array_split(y, num_folds)

	sum_train_acc = 0
	sum_val_acc = 0

	best_model = -1
	best_val_acc = -1

	for i in range(num_folds):
	   X_training_folds = []
	   y_training_folds = []
	   test_fold = i

	   for j in range(num_folds):
	       if i==j:
	           continue
	       X_training_folds.append(X_folds[j])
	       y_training_folds.append(y_folds[j])


	   X_train = np.concatenate(X_training_folds)
	   y_train = np.concatenate(y_training_folds)
	   
	   X_test = X_folds[i]
	   y_test = y_folds[i]

	   ss = StandardScaler()
	   X_train = ss.fit_transform(X_train)
	   X_test = ss.transform(X_test)

	   if classifier_type == 0:
	   	train_score, val_score = DecisionTreeClassification(X_train, X_test, y_train, y_test, params)
	   else:
	   	train_score, val_score = GaussianNaiveBayesClassification(X_train, X_test, y_train, y_test)

	   if val_score > best_val_acc:
	   	best_val_acc = val_score
	   	best_model = i


	   sum_val_acc += val_score
	   sum_train_acc += train_score

	training_acc = sum_train_acc/num_folds
	validation_acc = sum_val_acc/num_folds

	if save_best_model == True:
		X_training_folds = []
		y_training_folds = []
		test_fold = best_model

		for j in range(num_folds):
			if j==best_model:
				continue
			X_training_folds.append(X_folds[j])
			y_training_folds.append(y_folds[j])

		X_train = np.concatenate(X_training_folds)
		y_train = np.concatenate(y_training_folds)

		X_test = X_folds[test_fold]
		y_test = y_folds[test_fold]

		if classifier_type == 0:
			train_score, val_score = DecisionTreeClassification(X_train, X_test, y_train, y_test, params, True, dataset)
		else:
			train_score, val_score = GaussianNaiveBayesClassification(X_train, X_test, y_train, y_test, True, dataset)

	return training_acc, validation_acc

def GridSearch(X, y, classifier_type, params):

	"""
	Performs grid search on the given parameters.

	The classifier_type is used to denote the type of classifier to be used.
	0 is for DT, rest of the values for GNB.

	Parameters are provided as dictionary with list values.

	For e.g. params = {"random_state": random_seed, "max_depth": optimal_depth}

	"""
	list_of_lists = []
	for key in params:
		list_of_lists.append(params[key])

	parameters_list = list(product(*list_of_lists))
	
	train_acc_list = []
	val_acc_list = []

	best_paramaters_dict = {}
	best_validation_acc = -1
	
	for i in range(len(parameters_list)):
		parameters_dict = {}
		j = 0
		for key in params:
			parameters_dict[key] = parameters_list[i][j]
			j += 1

		print("Parameters:")
		print(parameters_dict)
		training_acc, validation_acc = KfoldCV(X, y, classifier_type = 0, params = parameters_dict)
		print("validation_acc:", validation_acc, end = "\n\n")

		train_acc_list.append(training_acc)
		val_acc_list.append(validation_acc)

		if validation_acc > best_validation_acc:
			best_validation_acc = validation_acc
			best_parameters_dict = parameters_dict

	print("Best validation_acc is: " + str(best_validation_acc) + " on Parameters: ", best_parameters_dict)

	plt.plot(list(range(1, len(train_acc_list)+1)), train_acc_list, label = "Training Accuracy")
	plt.plot(list(range(1, len(train_acc_list)+1)), val_acc_list, label = "Validation Accuracy")
	plt.xlabel("max_depth")
	plt.legend()
	plt.show()

def EvalutateTestingData(X_train, X_test, y_train, y_test, y_pred, clf, max_depth = 11):
	"""
	Evaluates the given input data based on the provided model.

	The accuracy scores, micro/macro precision, micro/macro recall,
	micro/macro f1 score, confused matrix are calculated and printed.

	Roc curve (both for multiclass and binary class data) is also plotted.

	The function returns the confusion matrix.

	"""
	
	confusion_matrix = []

	n = len(set(y_test))

	for i in range(n):
		a = []
		for j in range(n):
			a.append(0)
		confusion_matrix.append(a)


	for i in range(len(y_test)):
		confusion_matrix[y_test[i]][y_pred[i]] += 1

	confusion_matrix = np.array(confusion_matrix)

	print("\nConfusion Matrix:", end = "\n\n")
	print(confusion_matrix)

	accuracy = 0
	for i in range(n):
		accuracy += confusion_matrix[i][i]
	accuracy /= len(y_test)

	print("\nAccuracy:", accuracy)

	precision_list = []

	for j in range(n):
		correct = confusion_matrix[j][j]
		total = 0
		for i in range(n):
			total += confusion_matrix[i][j]
		precision_list.append(correct/total)

	precision_sum = 0
	for c in precision_list:
		precision_sum += c
	macro_precision = precision_sum/n
	print("\nMacro Precision:", macro_precision)

	recall_list = []

	for i in range(n):
		correct = confusion_matrix[i][i]
		total = 0
		for j in range(n):
			total += confusion_matrix[i][j]
		recall_list.append(correct/total)

	recall_sum = 0
	for c in recall_list:
		recall_sum += c
	macro_recall = recall_sum/n
	print("Macro Recall:", macro_recall)

	macro_f1 = 2*(macro_precision*macro_recall)/(macro_precision+macro_recall)
	print("Macro F1:", macro_f1)

	print("\nMicro Precision:", accuracy)
	print("Micro Recall:", accuracy)
	print("Micro F1:", accuracy)


	probs_all = clf.predict_proba(X_test)


	for i in range(n):
		if i == 0 and n==2:
			continue
		

		y_test_i = []

		for j in range(len(y_test)):
			if y_test[j] == i:
				y_test_i.append(1)
			else:
				y_test_i.append(0)

		
		probs = []
		for j in range(len(probs_all)):
			probs.append(probs_all[j][i])


		threshold_list = []

		for j in range(0, 1001):
			threshold_list.append(j*0.001)

		threshold_list.append(2)
		
		tpr_list = []
		fpr_list = []

		for threshold in threshold_list:
			y_pred_i = []
			for j in range(len(probs)):
				if probs[j] >= threshold:
					y_pred_i.append(1)
				else:
					y_pred_i.append(0)
			
			TP = 0
			TN = 0
			FP = 0
			FN = 0
			for j in range(len(y_test_i)):

				if y_pred_i[j] == 1:
					if y_test_i[j] == 1:
						TP += 1
					else:
						FP += 1
				elif y_pred_i[j] == 0:
					if y_test_i[j] == 1:
						FN += 1
					else:
						TN += 1

			TPR = TP/(TP+FN)
			FPR = FP/(FP+TN)

			tpr_list.append(TPR)
			fpr_list.append(FPR)

		combined_list = []

		for j in range(len(fpr_list)):
			combined_list.append((fpr_list[j], tpr_list[j]))

		combined_list.sort()
		fpr_list = [0]
		tpr_list = [0]

		for c in combined_list:
			fpr_list.append(c[0])
			tpr_list.append(c[1])
		
		plt.plot(fpr_list, tpr_list, label = "Class " + str(i))
	
	plt.xlabel("FPR")
	plt.ylabel("TPR")
	plt.title("Roc Curve")
	plt.legend()
	plt.show()
	plt.close()

	# return confusion_matrix

def DoAllParts(dataset  = "A"):
	"""
	Performs all the things required by the 
	assignment for the given dataset one by one
	"""


	print("Using Dataset", dataset)

	X, y = ReadDatasets(dataset)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

	print("Applying grid search to find optimal max depth for DT\n")
	GridSearch(X_train, y_train, 0, {"random_state":[random_seed], "max_depth": range(1,21)})

	if dataset == "A":
		optimal_depth = 11
	else:
		optimal_depth = 11

	print("\nGrid Search gives optimal max depth:", optimal_depth)

	print("Applying KfoldCV using DT classifier and max_depth " + str(optimal_depth) +  " (on 4 number of folds")
	train_acc, val_acc =  KfoldCV(X_train, y_train, num_folds = 4, classifier_type = 0, params = {"random_state": random_seed, "max_depth": optimal_depth}, save_best_model = True, dataset = dataset)
	print("DT: Training accuracy:", train_acc, ", Validation accuracy", val_acc)

	print("\nApplying KfoldCV using GNB classifier(on 4 number of folds")
	train_acc, val_acc =  KfoldCV(X_train, y_train, num_folds = 4, classifier_type = 1, save_best_model = True, dataset = dataset)
	print("GNB: Training accuracy:", train_acc, ", Validation accuracy", val_acc)


	print("\nClearly best bodel is DT on depth", optimal_depth)
	# print("Running KfoldCV again and saving the best model")
	# KfoldCV(X_train, y_train, num_folds = 4, classifier_type = 0, params = {"random_state": random_seed, "max_depth": optimal_depth}, save_best_model = True, dataset = dataset)

	print("Loading best model")

	f = open("dt_model_" + dataset + ".pkl", "rb")
	clf = pickle.load(f)
	f.close()

	test_acc = clf.score(X_test, y_test)
	y_pred = clf.predict(X_test)
	print("\nUsing DT,Test accurancy:", test_acc)

	print("\nEvaluating")
	EvalutateTestingData(X_train, X_test, y_train, y_test, y_pred, clf, max_depth = optimal_depth)

	f = open("gnb_model_" + dataset + ".pkl", "rb")
	clf = pickle.load(f)
	f.close()

	test_acc = clf.score(X_test, y_test)
	y_pred = clf.predict(X_test)
	print("\nUsing GNB,Test accurancy:", test_acc)

	print("\nEvaluating")
	EvalutateTestingData(X_train, X_test, y_train, y_test, y_pred, clf, max_depth = optimal_depth)

DoAllParts("A")
print("---------------------------------------------")
print("---------------------------------------------")
DoAllParts("B")
