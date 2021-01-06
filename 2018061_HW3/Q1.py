import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import pickle
import matplotlib.pyplot as plt

class MyNeuralNetwork():
    """
    My implementation of a Neural Network Classifier.
    """

    acti_fns = ['relu', 'sigmoid', 'linear', 'tanh', 'softmax']
    weight_inits = ['zero', 'random', 'normal']

    def __init__(self, n_layers, layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs):
        """
        Initializing a new MyNeuralNetwork object

        Parameters
        ----------
        n_layers : int value specifying the number of layers

        layer_sizes : integer array of size n_layers specifying the number of nodes in each layer

        activation : string specifying the activation function to be used
                                 possible inputs: relu, sigmoid, linear, tanh

        learning_rate : float value specifying the learning rate to be used

        weight_init : string specifying the weight initialization function to be used
                                    possible inputs: zero, random, normal

        batch_size : int value specifying the batch size to be used

        num_epochs : int value specifying the number of epochs to be used
        """ 
        self.n_layers = n_layers
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_init = weight_init
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weights = None
        self.biases = None
                        
        
        activation_fn_mapping = { 'relu': self.relu, 'sigmoid': self.sigmoid, 'linear': self.linear, 'tanh': self.tanh, 'softmax': self.softmax}
        derivative_fn_mapping = { 'relu': self.relu_grad, 'sigmoid': self.sigmoid_grad, 'linear': self.linear_grad, 'tanh': self.tanh_grad, 'softmax': self.softmax_grad}
        weight_init_mapping = {'zero': self.zero_init, 'random': self.random_init, 'normal': self.normal_init}
        
        self.activation_fn = activation_fn_mapping[activation]
        self.ac_derivation_fn = derivative_fn_mapping[activation]
        self.weight_init_fn = weight_init_mapping[weight_init]
        
        np.random.seed(seed=10)

        if activation not in self.acti_fns:
                raise Exception('Incorrect Activation Function')

        if weight_init not in self.weight_inits:
                raise Exception('Incorrect Weight Initialization Function')
        pass

    
    def relu(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """

        return X * (X>=0)


    def relu_grad(self, X):
        """
        Calculating the gradient of ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        
        return 1 * (X>0)


    def sigmoid(self, X):
        """
        Calculating the Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------probs = nn.predict_proba(X_test)
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1/(1 + np.exp(-X)) 


    def sigmoid_grad(self, X):
        """
        Calculating the gradient of Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return self.sigmoid(X)*(1-self.sigmoid(X))

    def linear(self, X):
        """
        Calculating the Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return X

    def linear_grad(self, X):
        """
        Calculating the gradient of Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.ones(X.shape)

    def tanh(self, X):
        """
        Calculating the Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 2/(1+(np.exp(-X)*np.exp(-X)))-1
                            
    def tanh_grad(self, X):
        """
        Calculating the gradient of Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1 - self.tanh(X)*self.tanh(X)
    
 
    
    def softmax(self, X):
        """
        Calculating the softmax activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        expX = np.exp(X - np.max(X))
        return expX / expX.sum(axis=0, keepdims=True)
                                        
    def softmax_grad(self, X):
        """
        Calculating the gradient of Softmax activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return softmax(X)*(1-softmax(X))

    def zero_init(self, shape):
        """
        Calculating the initial weights after Zero Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """

        return np.zeros(shape)

    def random_init(self, shape):
        """
        Calculating the initial weights after Random Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        return 0.01 * np.random.rand(shape[0], shape[1])

    def normal_init(self, shape):
        """
        Calculating the initial weights after Normal(0,1) Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        return  0.01 * np.random.randn(shape[0], shape[1])

    def fit(self, X, y, plot = False, X_test = None, y_test = None):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """

        # fit function has to return an instance of itself or else it won't work with test.py
        
        num_labels = len(np.unique(y))
        new_y = []
        for c in y:
            a = []
            for i in range(num_labels):
                a.append(0)
            a[c] = 1
            a = np.array(a)
            new_y.append(a)
        new_y = np.array(new_y)
        y = new_y

        if plot == True:
            new_y = []
            for c in y_test:
                a = []
                for i in range(num_labels):
                    a.append(0)
                a[c] = 1
                a = np.array(a)
                new_y.append(a)
            new_y = np.array(new_y)
            y_test = new_y
            
        n_rows, n_cols = X.shape
        self.weights = [0]
        self.biases = [0]
        
        training_loss_list = []
        validation_loss_list = []

        for i in range(0, self.n_layers -1):
            # initialize weights
            self.weights.append(self.weight_init_fn((self.layer_sizes[i], self.layer_sizes[i+1])))
            self.biases.append(np.zeros((1, self.layer_sizes[i+1])))
                                                                                                         
        for i in tqdm(range(self.num_epochs)): 
                
            new_X = copy.deepcopy(X)
            indices = np.array(list(range(0, len(X))))
            np.random.shuffle(indices)
            
            if len(X)%self.batch_size == 0:
                num_batches = len(X)//self.batch_size
            else:
                num_batches = len(X)//self.batch_size + 1
            
            tot_training_loss = 0
            
            for j in range(num_batches):
                # create batches

                X_batch = X[indices[j*self.batch_size: (j+1)*self.batch_size], :]
                y_batch = y[indices[j*self.batch_size: (j+1)*self.batch_size], :]
                A_list, Z_list = self.forward(X_batch)
                cost = -np.mean(y_batch* np.log(A_list[-1].T))
                tot_training_loss += cost
                self.backward(X_batch, y_batch , A_list, Z_list)

            if plot == True:
                # code for plotting loss graph
                A_list_t, Z_list_t = self.forward(X_test)
                cost = -np.mean(y_test* np.log(A_list_t[-1].T))
                validation_loss = cost  
            else:
                validation_loss = 0       
                    
            training_loss    = tot_training_loss/num_batches
            training_loss_list.append(training_loss)
            validation_loss_list.append(validation_loss)

            print("Training Loss:", training_loss)
                
        if plot == True:
            # code for plotting loss graph
            plt.plot(list(range(1, self.num_epochs+1)), training_loss_list, label = "Training Cross Entropy Loss")
            plt.plot(list(range(1, self.num_epochs+1)), validation_loss_list, label = "Validation Cross Entropy Loss")
            plt.legend()
            plt.title(self.activation + " : cross entropy loss vs epochs")
            plt.xlabel("No. of Epochs")
            plt.show()

        return self

                 
                    
    def predict_proba(self, X):
        """
        Predicting probabilities using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 2-dimensional numpy array of shape (n_samples, n_classes) which contains the 
                class wise prediction probabilities.
        """

        # return the numpy array y which contains the predicted values
        A_list, Z_list = self.forward(X)
        return A_list[-1].T

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        # return the numpy array y which contains the predicted values
        
        A_list, Z_list = self.forward(X)
        preds = np.argmax(A_list[-1], axis = 0)
        return preds

    def score(self, X, y):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as testing labels.

        Returns
        -------
        acc : float value specifying the accuracy of the model on the provided testing set
        """
        
        preds = self.predict(X)
        temp = (preds == y)
        
        return temp.sum()/len(X)


    def forward(self, X):
        """
        Applies forawrd pass for each of the layers of the neural network.
        Takes input X and return 2 lists
        A_list and Z_list the activations and WX + B
        """
            
        A_list = []
        A = X.T
        A_list.append(A)
        Z_list = [0]

        for i in range(self.n_layers - 2):
            Z = np.matmul(self.weights[i + 1].T, A) + self.biases[i + 1].T
            A = self.activation_fn(Z)
            A_list.append(A)
            Z_list.append(Z)

        Z = np.matmul(self.weights[-1].T, A) + self.biases[-1].T
        
        A = self.softmax(Z)
        A_list.append(A)
        Z_list.append(Z)

        return A_list, Z_list


    def backward(self, X, Y, A_list, Z_list):
        """
        Applies backwards pass algorithm and updates the weights and
        biases of the the model.
        The function takes input X, Y, A_list, Z_list
        The function does not return anything
        """

        derivatives_w = []
        derivatives_b = []
        
        for i in range(self.n_layers):
            derivatives_w.append(0)
            derivatives_b.append(0)
         

        A = A_list[-1]
        
        dz = A - Y.T

        dw = dz.dot(A_list[-2].T) / len(X)
        db = np.sum(dz, axis=1, keepdims=True) / len(X)
        daprev = self.weights[-1].dot(dz)
        
        derivatives_w[-1] = dw
        derivatives_b[-1] = db

        for i in range(self.n_layers - 2, 0, -1):
            dz = daprev * self.ac_derivation_fn(Z_list[i])
            dw = dz.dot(A_list[i-1].T)/len(X)
            db = np.sum(dz, axis=1, keepdims=True)/len(X)
            
            derivatives_w[i] = dw
            derivatives_b[i] = db
            if i == 1:
                continue
            daprev = self.weights[i].dot(dz)
        
        for i in range(1, self.n_layers):
            self.weights[i] -= self.learning_rate * derivatives_w[i].T
            self.biases[i] -= self.learning_rate * derivatives_b[i].T

    def save_weights(self):
        """
        Saves the weights and biases of the model to a pickle file.
        """
        f = open("weights_"+self.activation+".pkl", "wb")
        pickle.dump(self.weights, f)
        f.close()
        f = open("biases_"+self.activation+".pkl", "wb")
        pickle.dump(self.biases, f)
        f.close()
                

    

