import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy
import seaborn as sns

np.random.seed(0)

class MyPreProcessor():
    """
    My steps for pre-processing for the three datasets.
    """

    def __init__(self):
        pass

    def pre_process(self, dataset):
        """
        Reading the file and preprocessing the input and output.
        Note that you will encode any string value and/or remove empty entries in this function only.
        Further any pre processing steps have to be performed in this function too. 

        Parameters
        ----------

        dataset : integer with acceptable values 0, 1, or 2
        0 -> Abalone Dataset
        1 -> VideoGame Dataset
        2 -> BankNote Authentication Dataset
        
        Returns
        -------
        X : 2-dimensional numpy array of shape (n_samples, n_features)
        y : 1-dimensional numpy array of shape (n_samples,)
        """

        # # np.empty creates an empty array only. You have to replace this with your code.
        # X = np.empty((0,0))
        # y = np.empty((0))

        if dataset == 0:
            # Implement for the abalone dataset

            # Reading dataset csv into a pandas dataframe and giving names to columns
            df = pd.read_csv("Dataset.data", sep = " ", names = ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"])
            
            # dropping rows in dataframes with null values
            df.dropna(inplace = True)

            # Since sex was a categorical variable, I removed the original column and instead of that
            # added 3 columns: sex_m, sex_f, sex_i containing binary values and indicating if the species
            # had that particular gender or not
            sex_values = df['sex'].values
            sex_m = np.zeros([len(df), 1])
            sex_f = np.zeros([len(df), 1])
            sex_i = np.zeros([len(df), 1])
            for i in range(len(df)):
                if df['sex'][i] == 'M':
                    sex_m[i] = 1
                elif df['sex'][i] == 'F':
                    sex_f[i] = 1
                else:
                    sex_i[i] = 1
            del(df["sex"])
            df.insert(0, "sex_m", sex_m)
            df.insert(1, "sex_f", sex_f)
            df.insert(2, "sex_i", sex_i)

            # shuffling the rows of the dataframe using the seed, random_state = 0, and resetting indexes
            # of the rows since rows were dropped from it
            df = df.sample(frac = 1, random_state = 0).reset_index(drop = True) 

            # separating dataset into X and y
            X = df[df.columns[:-1]].values
            y = df[[df.columns[-1]]].values

            # converting y from Nx1 np array to 1D np array
            y = y.reshape(-1)

        elif dataset == 1:
            # Implement for the video game dataset

            # Reading dataset csv into a pandas dataframe
            df = pd.read_csv("VideoGameDataset - Video_Games_Sales_as_at_22_Dec_2016.csv")
            
            # only keeping the 3 relevant columns in the dataframe
            df = df[["Critic_Score", "User_Score", "Global_Sales"]]

            # dropping rows in dataframes with null values
            df.dropna(inplace = True)

            # Some global_sales value were given as tbd, removing all rows with such values
            df = df[df.User_Score != "tbd"]

            # converting every value in dataframe to float
            df["Critic_Score"] = df["Critic_Score"].astype('float64')
            df["User_Score"] = df["User_Score"].astype('float64')
            df["Global_Sales"] = df["Global_Sales"].astype('float64')

            # shuffling the rows of the dataframe using the seed, random_state = 0, and resetting indexes
            # of the rows since rows were dropped from it
            df = df.sample(frac = 1, random_state = 0).reset_index(drop = True) 
            
            # separating dataset into X and y
            X = df[df.columns[:-1]].values  
            y = df[[df.columns[-1]]].values
            
            # converting y from Nx1 np array to 1D np array
            y = y.reshape(-1)


        elif dataset == 2:
            # Implement for the banknote authentication dataset

            # Reading dataset csv into a pandas dataframe and giving names to columns
            df = pd.read_csv("data_banknote_authentication.txt", names = ["variance", "skewness", "curtosis", "entropy", "class"])
            
            # dropping rows in dataframes with null values
            df.dropna(inplace = True)

            # shuffling the rows of the dataframe using the seed, random_state = 0, and resetting indexes
            # of the rows since rows were dropped from it
            df = df.sample(frac = 1, random_state = 0).reset_index(drop = True) 

            # separating dataset into X and y
            X = df[df.columns[:-1]].values
            y = df[[df.columns[-1]]].values

            # converting y from Nx1 np array to 1D np array
            y = y.reshape(-1)


        
        # normalising X
        X = (X-X.mean(axis = 0))/X.std(axis = 0)

        # Adding a column of ones to X to take care of the constant term(bias) in theta
        num_rows, num_cols = X.shape
        new_column = np.ones([num_rows, 1])
        X = np.append(X, new_column, axis = 1)

        return X, y

class MyLinearRegression():
    """
    My implementation of Linear Regression.
    """

    def __init__(self):
        """
        two class variables:
        1) parameters obtained in the last fit call
        2) list of parameters iteration wise, obtained in the last fit call
        """
        self.parameters = None
        self.parameters_iterations_wise = []

    def fit(self, X, y, loss_type = 0, dataset = 0):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        loss_type: indicates loss_type. 0 means RMSE loss, 1 means MAE loss.

        dataset: indicated index of dataset on which the model is applied

        Returns
        -------
        self : an instance of self
        """
        num_rows, num_cols = X.shape

        # Initializing parameters as a Nx1 np array of zeroes
        parameters = np.zeros([num_cols, 1])
        
        # assigning values to learning_rate and num_iterations according the dataset
        if dataset == 0:
            learning_rate = 0.01
            num_iterations = 1000
        else:
            learning_rate = 0.00001
            num_iterations = 1000

        # Calling gradient descent and updating paramters
        for i in range(num_iterations):
            if loss_type == 0:
                # Calling function to calcucalate gradient according to RMSE loss, and update parameters
                parameters = self.update_parameters_rmse(parameters, X, y, learning_rate)
            else:
                # Calling function to calcucalate gradient according to MAE loss, and update parameters
                parameters = self.update_parameters_mae(parameters, X, y, learning_rate)
            
            # adding new updated parameter to the list 
            self.parameters_iterations_wise.append(copy.deepcopy(parameters))

        # updating instance attribute parameter variable
        self.parameters = parameters

        # fit function has to return an instance of itself or else it won't work with test.py
        return self

    def predict(self, X, parameters = None):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.
    
        parameters: the caller may provide theta values for prediction. If they are not provided, 
        parameters obtained in the last fit call are used.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        # return the numpy array y which contains the predicted values
        num_rows, num_cols = X.shape

        if parameters is None:
            # If they are not provided, 
            # parameters obtained in the last fit call are used
            parameters = self.parameters

        # obtaining predicted values
        y = np.matmul(X, parameters)

        # converting obtained y array to 1d array
        y = y.reshape(-1)
        
        return y

    def update_parameters_rmse(self, parameters, X, y, learning_rate):
        """ 
        updates parameters after calculating gradient using rmse loss function

        returns updated parameters in np array format
        """
        j_theta = 0
        num_rows, num_cols = X.shape
        
        #calculating predicted values 
        pred_values = np.matmul(X, parameters)
        pred_values = pred_values.reshape(-1)
        
        # making another np array which stores (predicted value - actual value)
        pred_minus_yi = np.subtract(pred_values, y)
    
        # Calculating loss
        j_theta = np.sum(pred_minus_yi**2)
        j_theta = j_theta**0.5
        
        X_transpose = np.transpose(X)
        
        # Calculating gradient term for each parameter
        derivative_array = np.matmul(X_transpose, pred_minus_yi) 

        # updating parameter values
        for j in range(num_cols):
            parameters[j][0] -= (2*derivative_array[j]*learning_rate)/(j_theta*(num_rows**0.5)*2)
        
        return parameters

    def update_parameters_mae(self, parameters, X, y, learning_rate):
        """ 
        updates parameters after calculating gradient using mae loss function

        returns updated parameters in np array format
        """
        num_rows, num_cols = X.shape
        
        pred_values = [] 
        # calculating predicted values according to current parameters
        pred_values = np.matmul(X, parameters)
        pred_values = pred_values.reshape(-1)
        
        # making another np array which stores (predicted value - actual value)
        pred_minus_yi = np.subtract(pred_values, y)

        # Storeing signum of pred_minus_yi array
        signum_vals = np.sign(pred_minus_yi)
        

        # Storeing gradient term for each parameter
        X_transpose = np.transpose(X)
        derivative_array = np.matmul(X_transpose, signum_vals) 

        # updating parameter values
        for j in range(num_cols):
            parameters[j][0] -= (derivative_array[j]*learning_rate)/num_rows

        return parameters

    def KfoldCV(self, X, y, num_folds, loss_type = 0, dataset = 0, flag = 0, plot = False):
        """
        Runs kfold cross validation with K = num_folds.
        Loss_type parameter indicated which loss function to use.
        Dataset parameter tells which dataset to use.
        If flag == 0: function returns mean error on all folds, else
        it returns the split which gives the minimum validation loss.

        If plot flag it True, function also plots the training loss/validation
        loss on the split which gives the minimum validation loss
        """

        num_rows, num_cols = X.shape

        # dividing X and y into folds
        X_folds = np.array_split(X, num_folds)
        y_folds = np.array_split(y, num_folds)

        sum_error = 0

        min_validation_loss = 99999999
        best_parameters_iteration_wise = []
        best_split = -1
        best_Xtrain = None
        best_ytrain = None
        best_Xtest = None
        best_ytest = None

        # running num_folds times, taking the ith fold as the
        # testing fold each time
        for i in range(num_folds):
            # storeing the folds as a list
            X_training_folds = []
            y_training_folds = []
            test_fold = i

            # adding folds to training folds list
            for j in range(num_folds):
                if i==j:
                    continue
                X_training_folds.append(X_folds[j])
                y_training_folds.append(y_folds[j])


            # creates training sets from list of np arrays
            Xtrain = np.concatenate(X_training_folds)
            ytrain = np.concatenate(y_training_folds)
            
            Xtest = X_folds[i]
            ytest = y_folds[i]

            # storing parameters for every iteration in a list
            self.parameters_iterations_wise = []

            # fitting the model on training sets
            self.fit(Xtrain, ytrain, loss_type, dataset)
            
            # predicting values on the testing set
            ypred = self.predict(Xtest)
        
            validation_loss = self.CalculateError(ypred, ytest, loss_type)

            print("Split: " + str(i+1) + ", Validation loss: " + str(validation_loss))

            # checking if validation loss is less than that of the
            # best split obtained till now, if it is storing it in these variables
            if validation_loss < min_validation_loss:
                min_validation_loss = validation_loss
                best_parameters_iteration_wise = copy.deepcopy(self.parameters_iterations_wise)
                
                best_split = i
                best_Xtrain = Xtrain
                best_ytrain = ytrain
                best_Xtest = Xtest
                best_ytest = ytest


            sum_error += validation_loss

        # Calculating training loss/validation loss iteration wise 
        # on the split with the minimum validation loss
        training_loss_iteration_wise = []
        validation_loss_iteration_wise = []
        for parameters in best_parameters_iteration_wise:
            ypred = self.predict(best_Xtest, parameters)
            validation_loss = self.CalculateError(ypred, best_ytest, loss_type)

            ypred_train = self.predict(best_Xtrain, parameters)
            training_loss = self.CalculateError(ypred_train, best_ytrain, loss_type)

            training_loss_iteration_wise.append(training_loss)
            validation_loss_iteration_wise.append(validation_loss)

        # Calculating mean validation loss of all splits
        error = sum_error/num_folds

        print("Number of folds: " + str(num_folds) + ", Mean error: " + str(error) + ", best split: " + str(best_split+1))

        if plot == True:
            # plotting training loss/validation loss vs iterations

            plt.plot(training_loss_iteration_wise, label = "Training Loss")
            plt.plot(validation_loss_iteration_wise, label = "Validation Loss")
            plt.xlabel("No. of Iterations")
            plt.legend()
            plt.show()

        if flag == 0:
            return error
        else:
            return best_Xtrain, best_ytrain, best_Xtest, best_ytest


    def CallKfoldCV(self, X, y, loss_type = 0, dataset = 0):
        """ 
        Calls function to run k fold CV for different value of K
        starting from 3 to 10. Reports the value of k
        which gives the least mean validation Loss
            
        """
        min_error = -1
        best_num_folds = 100

        # running for values of K from 3 to 10
        for i in range(3, 11):
            error = self.KfoldCV(X, y, i, loss_type, dataset, 0, False)
            if min_error == -1 or error < min_error:
                best_num_folds = i
                min_error = error
       
        print("Best mean loss is " + str(min_error) + " on " + str(best_num_folds) + " number of folds")


        

    def CalculateError(self, ypred, ytest, loss_type = 0):
        """
        Takes input the predicted and the actual values and the loss function
        and returns the total Testing loss
        """

        # creating np array which store predicted minus actual values
        pred_minus_yi = np.subtract(ypred, ytest)
        
        if loss_type == 0:
            rmse = np.sum(pred_minus_yi**2)    
            rmse /= len(ypred)
            rmse = rmse**0.5
            error = rmse
        else:
            pred_minus_yi_abs = np.absolute(pred_minus_yi)
            mae = np.sum(pred_minus_yi_abs)
            mae /= len(ypred)
            error = mae
        return error

    def NormalFormParameters(self, X, y):
        """
        Takes input X and Y, and the function
        returns the optimal parameters calculated using the normal
        equation form 
        """
        first_term = np.linalg.inv(np.dot(np.transpose(X), X))
        second_term = np.dot(np.transpose(X), y)
        parameters = np.dot(first_term, second_term)
        return parameters.reshape((len(parameters),1))

    def Question1_partE(self, X, y, loss_type, num_folds):
        """
        Calculates the training and validation loss
        for the best fold obtained with a particular K values
        and a loss_type for the given dataset
        
        """
        num_rows, num_cols = X.shape
        
        # getting parameters using normal form
        parameters = self.NormalFormParameters(X, y)
        print("Optimal parameters:", parameters)

        # getting training and testing set for the best fold for thid number of folds
        Xtrain, ytrain, Xtest, ytest = self.KfoldCV(X, y, num_folds = num_folds, loss_type = loss_type, dataset = 0, flag = 1, plot = False)
        
        # predicting training and testing data using parameters
        ypred_train = self.predict(Xtrain, parameters)
        ypred_test = self.predict(Xtest, parameters)

        # calculating training and validation loss
        training_loss = self.CalculateError(ypred_train, ytrain, loss_type)
        validation_loss = self.CalculateError(ypred_test, ytest, loss_type)

        print("Training loss:", training_loss)
        print("Validation loss:", validation_loss)

class MyLogisticRegression():
    """
    My implementation of Logistic Regression.
    """

    def __init__(self):
        self.parameters = None
        self.parameters_iterations_wise = []

    def fit(self, X, y, type_GD = 0, learning_rate = 0.03, num_iterations = 10000):
        """
        Fitting (training) the logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
            
        type_GD: indicated type of GD to use. 0 for SGD and 1 for BGD>
        
        learning_rate and num_iterations can also be given as parameters to this function.
        Returns
        -------
        self : an instance of self
        """

        # fit function has to return an instance of itself or else it won't work with test.py
        
        num_rows, num_cols = X.shape

        parameters = np.zeros([num_cols, 1])

        # Calling gradient descent and update paramters
        for i in range(num_iterations):
            if type_GD == 0:
                # For SGD
                parameters = self.update_parameters_sgd(parameters, X, y, learning_rate)
            else:
                # FOR BGD
                parameters = self.update_parameters_bgd(parameters, X, y, learning_rate)
            
            # adding parameters to list stored as an instance attribute
            self.parameters_iterations_wise.append(copy.deepcopy(parameters))
             
        # updating instance attribute variable for parameters
        self.parameters = parameters

        # fit function has to return an instance of itself or else it won't work with test.py
        return self

    def predict(self, X, parameters = None):
        """
        Predicting values using the trained logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        # return the numpy array y which contains the predicted values

        if parameters is None:
            # if parameters are not given input 
            # taking parameters which are stored as instance attribute 
            parameters = self.parameters
        
        # predicting y
        y = np.matmul(X, parameters)
        
        # applying sigmoid function to obtained values
        # then replacing by 1 if value >=0.5, else by 0 
        y = 1/(1+np.exp(-y))
        y = (y >= 0.5)
        y = y.astype(int)
        
        return y

    def update_parameters_sgd(self, parameters, X, y, learning_rate):
        """ 
        updates parameters after calculating gradient using SGD

        returns updated parameters in np array format
        """

        num_rows, num_cols = X.shape
        
        # Getting a random row index
        RandomVal = np.random.randint(0, num_rows)

        # Calculating gradient for this chosen row
        pred_value = np.matmul(X[RandomVal], parameters)
        hypothesis = 1/(1+np.exp(-pred_value))
        ChosenRow = X[RandomVal]
        ChosenRow = ChosenRow.reshape((num_cols, 1))

        # updating parameters
        parameters -= learning_rate*(hypothesis-y[RandomVal])*ChosenRow

        return parameters

    def update_parameters_bgd(self, parameters, X, y, learning_rate):
        """ 
        updates parameters after calculating gradient using BGD

        returns updated parameters in np array format
        """
        num_rows, num_cols = X.shape

        # predicting values for entire X"
        pred_values = np.matmul(X, parameters)
        
        # getting hypotheseis values for predicted values
        hypothesis_values = 1/(1+np.exp(-pred_values))
        y = y.reshape((num_rows, 1))

        # calculating gradient
        hypothesis_minus_yi = hypothesis_values - y
        X_transpose = np.transpose(X)
        derivative_array = np.matmul(X_transpose, hypothesis_minus_yi) 
        
        # updating paramters
        parameters -= (learning_rate/num_rows)*derivative_array

        return parameters

    def SplitRunAndPlot(self, X, y, type_GD = 0, learning_rate = 0.01, num_iterations = 1000):
        """ Divides given dataset to 7:1:2 ratio for training:validation:testing set
        and SGD or BGD according to the value of type_GD,
        and then plot the training and validation losses.

        """

        # creating training set
        X_folds = np.array_split(X, 10)
        y_folds = np.array_split(y, 10)

        X_training_folds = []
        y_training_folds = []

        for i in range(7):
            X_training_folds.append(X_folds[i])
            y_training_folds.append(y_folds[i])

        Xtrain = np.concatenate(X_training_folds)
        ytrain = np.concatenate(y_training_folds)

        # creating validation set
        Xvalidation = X_folds[7]
        yvalidation = y_folds[7]


        X_testing_folds = []
        y_testing_folds = []

        for i in range(8, 10):
            X_testing_folds.append(X_folds[i])
            y_testing_folds.append(y_folds[i])

        # creating testing set
        Xtest = np.concatenate(X_testing_folds)
        ytest = np.concatenate(y_testing_folds)


        
        self.parameters_iterations_wise = []
        # fitting model on training set
        self.fit(Xtrain, ytrain, type_GD = type_GD, learning_rate = learning_rate, num_iterations = num_iterations)

        training_loss_iteration_wise = []
        validation_loss_iteration_wise = []

        # getting training and validation losses for each iteration
        for parameters in self.parameters_iterations_wise:

            ypred = np.matmul(Xvalidation, parameters)
            hypothesis_validation = 1/(1+np.exp(-ypred))
            hypothesis_validation = hypothesis_validation.reshape(-1)
            validation_loss = self.CalculateLoss(hypothesis_validation, yvalidation)

            ypred_train = np.matmul(Xtrain, parameters)
            hypothesis_train = 1/(1+np.exp(-ypred_train))
            hypothesis_train = hypothesis_train.reshape(-1)
            training_loss = self.CalculateLoss(hypothesis_train, ytrain)

            training_loss_iteration_wise.append(training_loss)
            validation_loss_iteration_wise.append(validation_loss)


        # plotting the training and validation losses for each iteration
        plt.plot(training_loss_iteration_wise, label = "Training Loss")
        plt.plot(validation_loss_iteration_wise, label = "Validation Loss")
        plt.xlabel("No. of Iterations")
        plt.legend()
        plt.show()

        # calculating training accuracy
        ytrain_pred = self.predict(Xtrain)
        correct = 0
        total = 0
        for i in range(len(ytrain_pred)):
            if ytrain_pred[i] == ytrain[i]:
                correct += 1
            total += 1
        print("Training Accuracy:", (correct/total)*100)

        # calculatting testing accuracy
        ytest_pred = self.predict(Xtest)
        correct = 0
        total = 0
        for i in range(len(ytest_pred)):
            if ytest_pred[i] == ytest[i]:
                correct += 1
            total += 1
        print("Testing Accuracy:", (correct/total)*100)
    
    def CalculateLoss(self, hypothesis, y):
        """Returns cross entopy loss for the given
        predicted and actual values"""

        num_rows = len(hypothesis)
        ones_array = np.ones(num_rows)
        one_minus_hypothesis = ones_array-hypothesis
        one_minus_yi = ones_array-y

        # Calculating first term of equation
        first_term_array = np.multiply(y, np.log(10**(-6)+hypothesis))
        
        # Calculating first term of equation
        second_term_array = np.multiply(one_minus_yi, np.log(10**(-6)+one_minus_hypothesis))
        
        # Adding both terms
        final_array = first_term_array+second_term_array

        # taking summation and then dividing
        loss = np.sum(final_array)
        loss = (-1*loss)/num_rows

        return loss

def EDA():
    # reading the dataset into pandas dataframe
    df = pd.read_csv("data_banknote_authentication.txt", names = ["variance", "skewness", "curtosis", "entropy", "class"])
    
    # dropping rows with null values
    df.dropna(inplace = True)


    # getting class variable distribution information
    print("Class variable distribution information:")
    print(df['class'].describe())

    # getting histogram for the dataframe
    df.hist()
    plt.show()

    # getting correlation matrix for the dataframe
    print("\nCorrelation Matrix: ")
    correlation_matrix = df.corr()
    print(correlation_matrix)

    print("\nCorrelation column only to class variable")
    correlation_to_class = correlation_matrix['class']
    print(correlation_to_class)

    # getting pairplot for the dataframe 
    print("Pairplot:")
    sns.pairplot(df,hue='class',diag_kind="hist")
    plt.show()