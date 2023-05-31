import numpy as np
import optimizers as opt    

#from keras import optimizers as opt
import sys  # for sys.float_info.epsilon

######################################################################
## class NeuralNetwork()
######################################################################




class NeuralNetwork():

    """
    A class that represents a neural network for nonlinear regression.

    Attributes
    ----------
    n_inputs : int
        The number of values in each sample
    n_hidden_units_by_layers : list of ints, or empty
        The number of units in each hidden layer.
        Its length specifies the number of hidden layers.
    n_outputs : int
        The number of units in output layer
    all_weights : one-dimensional numpy array
        Contains all weights of the network as a vector
    Ws : list of two-dimensional numpy arrays
        Contains matrices of weights in each layer,
        as views into all_weights
    all_gradients : one-dimensional numpy array
        Contains all gradients of mean square error with
        respect to each weight in the network as a vector
    Grads : list of two-dimensional numpy arrays
        Contains matrices of gradients weights in each layer,
        as views into all_gradients
    total_epochs : int
        Total number of epochs trained so far
    performance_trace : list of floats
        Mean square error (unstandardized) after each epoch
    n_epochs : int
        Number of epochs trained so far
    X_means : one-dimensional numpy array
        Means of the components, or features, across samples
    X_stds : one-dimensional numpy array
        Standard deviations of the components, or features, across samples
    T_means : one-dimensional numpy array
        Means of the components of the targets, across samples
    T_stds : one-dimensional numpy array
        Standard deviations of the components of the targets, across samples
    debug : boolean
        If True, print information to help with debugging
        
    Methods
    -------
    make_weights_and_views(shapes)
        Creates all initial weights and views for each layer

    train(X, T, n_epochs, method='sgd', learning_rate=None, verbose=True)
        Trains the network using input and target samples by rows in X and T

    use(X)
        Applies network to inputs X and returns network's output
    """

    def __init__(self, n_inputs, n_hiddens_each_layer, n_outputs):
        """Creates a neural network with the given structure

        Parameters
        ----------
        n_inputs : int
            The number of values in each sample
        n_hidden_units_by_layers : list of ints, or empty
            The number of units in each hidden layer.
            Its length specifies the number of hidden layers.
        n_outputs : int
            The number of units in output layer

        Returns
        -------
        NeuralNetwork object
        """

        self.n_inputs = n_inputs
        self.n_hiddens_each_layer = n_hiddens_each_layer
        self.n_outputs = n_outputs

        # Create one-dimensional numpy array of all weights with random initial values

        shapes = []
        n_in = n_inputs
        for nu in self.n_hiddens_each_layer + [n_outputs]:
            shapes.append((n_in + 1, nu))
            n_in = nu

        # Build list of views (pairs of number of rows and number of columns)
        # by reshaping corresponding elements from vector of all weights 
        # into correct shape for each layer.        

        self.all_weights, self.Ws = self.make_weights_and_views(shapes)
        self.all_gradients, self.Grads = self.make_weights_and_views(shapes)

        self.X_means = None
        self.X_stds = None
        self.T_means = None
        self.T_stds = None

        self.total_epochs = 0
        self.performance = None
        self.performance_trace = []
        self.debug = False
        
    def __repr__(self):
        return '{}({}, {}, {})'.format(type(self).__name__, self.n_inputs, self.n_hiddens_each_layer, self.n_outputs)

    def __str__(self):
        s = self.__repr__()
        if self.total_epochs > 0:
            s += '\n Trained for {} epochs.'.format(self.total_epochs)
            s += '\n Final standardized RMSE {:.4g}.'.format(self.performance_trace[-1])
        return s
 
    def make_weights_and_views(self, shapes):
        """Creates vector of all weights and views for each layer

        Parameters
        ----------
        shapes : list of pairs of ints
            Each pair is number of rows and columns of weights in each layer.
            Number of rows is number of inputs to layer (including constant 1).
            Number of columns is number of units, or outputs, in layer.

        Returns
        -------
        Vector of all weights, and list of views into this vector for each layer
        """

        # Make vector of all weights by stacking vectors of weights one layer at a time
        # Divide each layer's weights by square root of number of inputs
        all_weights = np.hstack([np.random.uniform(-1, 1, size=shape).flat / np.sqrt(shape[0])
                                 for shape in shapes])

        # Build weight matrices as list of views (pairs of number of rows and number 
        # of columns) by reshaping corresponding elements from vector of all weights 
        # into correct shape for each layer.  
        # Do the same to make list of views for gradients.
 
        views = []
        first_element = 0
        for shape in shapes:
            n_elements = shape[0] * shape[1]
            last_element = first_element + n_elements
            views.append(all_weights[first_element:last_element].reshape(shape))
            first_element = last_element

        # Set output layer weights to zero.
        views[-1][:] = 0
        
        return all_weights, views

    def set_debug(self, d):
        """Set or unset printing of debugging information.

        Parameters
        ----------
        d : boolean
            If True, print debugging information. 
        """
        
        self.debug = d
        if self.debug:
            print('Debugging information will now be printed.')
        else:
            print('No debugging information will be printed.')
        
    def train(self, X, T, n_epochs, method='sgd', learning_rate=None, verbose=True):
        """Updates the weights.

        Parameters
        ----------
        X : two-dimensional numpy array 
            number of samples  by  number of input components
        T : two-dimensional numpy array
            number of samples  by  number of output components
        n_epochs : int
            Number of passes to take through all samples
        method : str
            'sgd', 'adam', or 'scg'
        learning_rate : float
            Controls the step size of each update, only for sgd and adam
        verbose: boolean
            If True, progress is shown with print statements

        Returns
        -------
        self : NeuralNetwork instance
        """

        # Calculate and assign standardization parameters

        if self.X_means is None:
            self.X_means = X.mean(axis=0)
            self.X_stds = X.std(axis=0)
            self.X_stds[self.X_stds == 0] = 1
            self.T_means = T.mean(axis=0)
            self.T_stds = T.std(axis=0)

        # Standardize X and T.  Assign back to X and T.

        X = (X - self.X_means) / self.X_stds
        T = (T - self.T_means) / self.T_stds

        # Instantiate Optimizers object by giving it vector of all weights
        
        optimizer = opt.Optimizers(self.all_weights)

        # Define function to convert mean-square error to root-mean-square error,
        # Here we use a lambda function just to illustrate its use.  
        # We could have also defined this function with
        # def error_convert_f(err):
        #     return np.sqrt(err)

        error_convert_f = lambda err: np.sqrt(err)

        # Call the requested optimizer method to train the weights.

        if method == 'sgd':

            performance_trace = optimizer.sgd(self._error_f, self._gradient_f,
                                              fargs=[X, T], n_epochs=n_epochs,
                                              learning_rate=learning_rate,
                                              error_convert_f=error_convert_f,
                                              #error_convert_name='RMSE',
                                              verbose=verbose)

        elif method == 'adam':

            performance_trace = optimizer.adam(self._error_f, self._gradient_f,
                                               fargs=[X, T], n_epochs=n_epochs,
                                               learning_rate=learning_rate,
                                               error_convert_f=error_convert_f,
                                               #error_convert_name='RMSE',
                                               verbose=verbose)

        elif method == 'scg':

            performance_trace = optimizer.scg(self._error_f, self._gradient_f,
                                              fargs=[X, T], n_epochs=n_epochs,
                                              error_convert_f=error_convert_f,
                                              error_convert_name='RMSE',
                                              verbose=verbose)

        else:
            raise Exception("method must be 'sgd', 'adam', or 'scg'")

        self.total_epochs += len(performance_trace)
        self.performance_trace += performance_trace

        # Return neural network object to allow applying other methods
        # after training, such as:    Y = nnet.train(X, T, 100, 0.01).use(X)

        return self

    def _add_ones(self, X):
        return np.insert(X, 0, 1, 1)
    
    def _forward(self, X):
        """Calculate outputs of each layer given inputs in X.
        
        Parameters
        ----------
        X : input samples, standardized with first column of constant 1's.

        Returns
        -------
        Standardized outputs of all layers as list, include X as first element.
        """

        self.Zs = [X]

        # Append output of each layer to list in self.Zs, then return it.

        for W in self.Ws[:-1]:  # forward through all but last layer
            self.Zs.append(np.tanh(self._add_ones(self.Zs[-1]) @ W))
        last_W = self.Ws[-1]
        self.Zs.append(self._add_ones(self.Zs[-1]) @ last_W)

        return self.Zs

    # Function to be minimized by optimizer method, mean squared error
    def _error_f(self, X, T):
        """Calculate output of net given input X and its mean squared error.
        Function to be minimized by optimizer.

        Parameters
        ----------
        X : two-dimensional numpy array, standardized
            number of samples  by  number of input components
        T : two-dimensional numpy array, standardized
            number of samples  by  number of output components

        Returns
        -------
        Standardized mean square error as scalar float that is the mean
        square error over all samples and all network outputs.
        """

        if self.debug:
            print('in _error_f: X[0] is {} and T[0] is {}'.format(X[0], T[0]))
        Zs = self._forward(X)
        mean_sq_error = np.mean((T - Zs[-1]) ** 2)
        if self.debug:
            print(f'in _error_f: mse is {mean_sq_error}')
        return mean_sq_error

    # Gradient of function to be minimized for use by optimizer method
    def _gradient_f(self, X, T):
        """Returns gradient wrt all weights. Assumes _forward already called
        so input and all layer outputs stored in self.Zs

        Parameters
        ----------
        X : two-dimensional numpy array, standardized
            number of samples  x  number of input components
        T : two-dimensional numpy array, standardized
            number of samples  x  number of output components

        Returns
        -------
        Vector of gradients of mean square error wrt all weights
        """

        # Assumes forward_pass just called with layer outputs saved in self.Zs.

        if self.debug:
            print('in _gradient_f: X[0] is {} and T[0] is {}'.format(X[0], T[0]))
        n_samples = X.shape[0]
        n_outputs = T.shape[1]
        n_layers = len(self.n_hiddens_each_layer) + 1

        # delta is delta matrix to be back propagated.
        # Dividing by n_samples and n_outputs here replaces the scaling of
        # the learning rate.

        delta = -(T - self.Zs[-1]) / (n_samples * n_outputs)

        # Step backwards through the layers to back-propagate the error (delta)
        self._backpropagate(self, delta)

        return self.all_gradients

    def _backpropagate(self, delta):
        """Backpropagate output layer delta through all previous layers,
        setting self.Grads, the gradient of the objective function wrt weights in each layer.

        Parameters
        ----------
        delta : two-dimensional numpy array of output layer delta values
            number of samples  x  number of output components
        """

        n_layers = len(self.n_hiddens_each_layer) + 1
        if self.debug:
            print('in _gradient_f: first delta calculated is\n{}'.format(delta))
        for layeri in range(n_layers - 1, -1, -1):
            # gradient of all but bias weights
            self.Grads[layeri][:] = self._add_ones(self.Zs[layeri]).T @ delta
            # Back-propagate this layer's delta to previous layer
            if layeri > 0:
                delta = delta @ self.Ws[layeri][1:, :].T * (1 - self.Zs[layeri] ** 2)
                if self.debug:
                    print('in _gradient_f: next delta is\n{}'.format(delta))

    def use(self, X):
        """Return the output of the network for input samples as rows in X.
        X assumed to not be standardized.

        Parameters
        ----------
        X : two-dimensional numpy array
            number of samples  by  number of input components, unstandardized

        Returns
        -------
        Output of neural network, unstandardized, as numpy array
        of shape  number of samples  by  number of outputs
        """

        # Standardize X
        X = (X - self.X_means) / self.X_stds
        Zs = self._forward(X)
        # Unstandardize output Y before returning it
        return Zs[-1] * self.T_stds + self.T_means

    def get_performance_trace(self):
        """Returns list of unstandardized root-mean square error for each epoch"""
        return self.performance_trace


######################################################################
## class NeuralNetworkClassifier(NeuralNetwork)
######################################################################

class NeuralNetworkClassifier(NeuralNetwork):

    def __str__(self):
        s = self.__repr__()  # using NeuralNetwork.__repr__()
        if self.total_epochs > 0:
            s += '\n Trained for {} epochs.'.format(self.total_epochs)
            s += '\n Final data likelihood {:.4g}.'.format(self.performance_trace[-1])
        return s
 
    def _make_indicator_vars(self, T):
        """Convert column matrix of class labels (ints or strs) into indicator variables

        Parameters
        ----------
        T : two-dimensional array of all ints or all strings
            number of samples by 1
        
        Returns
        -------
        Two dimensional array of indicator variables. Each row is all 0's except one value of 1.
            number of samples by number of output components (number of classes)
        """

        # Make sure T is two-dimensional. Should be n_samples x 1.
        if T.ndim == 1:
            T = T.reshape((-1, 1))    
        return (T == np.unique(T)).astype(float)  # to work on GPU

    def _softmax(self, Y):
        """Convert output Y to exp(Y) / (sum of exp(Y)'s)

        Parameters
        ----------
        Y : two-dimensional array of network output values
            number of samples by number of output components (number of classes)

        Returns
        -------
        Two-dimensional array of indicator variables representing Y
            number of samples by number of output components (number of classes)
        """

        # Trick to avoid overflow
        # maxY = max(0, self.max(Y))
        maxY = Y.max()  #self.max(Y))        
        expY = np.exp(Y - maxY)
        denom = expY.sum(1).reshape((-1, 1))
        Y_softmax = expY / (denom + sys.float_info.epsilon)
        return Y_softmax

    # Function to be minimized by optimizer method, mean squared error
    def _neg_log_likelihood_f(self, X, T):
        """Calculate output of net given input X and the resulting negative log likelihood.
        Function to be minimized by optimizer.

        Parameters
        ----------
        X : two-dimensional numpy array, standardized
            number of samples  by  number of input components
        T : two-dimensional numpy array of class indicator variables
            number of samples  by  number of output components (number of classes)

        Returns
        -------
        Negative log likelihood as scalar float.
        """
        
        
        Ys = self._forward(X) 
        Y_softMax = self._softmax(Ys[-1])

        neg_mean_log_likelihood =  - np.mean(T * np.log(Y_softMax + sys.float_info.epsilon))

        return neg_mean_log_likelihood

    def _gradient_f(self, X, T):
        """Returns gradient wrt all weights. Assumes _forward (from NeuralNetwork class)
        has already called so input and all layer outputs stored in self.Zs

        Parameters
        ----------
        X : two-dimensional numpy array, standardized
            number of samples  x  number of input components
        T : two-dimensional numpy array of class indicator variables
            number of samples  by  number of output components (number of classes)

        Returns
        -------
        Vector of gradients of negative log likelihood wrt all weights
        """

        n_samples = X.shape[0]
        n_outputs = T.shape[1]

       
        D = -(T - self.Zs[-1]) / (n_samples * n_outputs)
        self._backpropagate(D)


        return self.all_gradients

    def train(self, X, T, n_epochs, method='sgd', learning_rate=None, verbose=True):
        """Updates the weights.

        Parameters
        ----------
        X : two-dimensional numpy array 
            number of samples  by  number of input components
        T : two-dimensional numpy array of target classes, as ints or strings
            number of samples  by  1
        n_epochs : int
            Number of passes to take through all samples
        method : str
            'sgd', 'adam', or 'scg'
        learning_rate : float
            Controls the step size of each update, only for sgd and adam
        verbose: boolean
            If True, progress is shown with print statements

        Returns
        -------
        self : NeuralNetworkClassifier instance
        """

        # Calculate and assign standardization parameters

        if self.X_means is None:
            self.X_means = X.mean(axis=0)
            self.X_stds = X.std(axis=0)
            self.X_stds[self.X_stds == 0] = 1
            # Not standardizing target classes.

        # Standardize X and assign back to X.

        X = (X - self.X_means) / self.X_stds

        # Assign class labels to self.classes, and counts of each to counts.
        # Create indicator values representation from target labels in T.

        self.classes, counts = np.unique(T, return_counts=True)
        T_ind_vars = self._make_indicator_vars(T) 

        # Instantiate Optimizers object by giving it vector of all weights.
        optimizer = opt.Optimizers(self.all_weights)

        # Define function to convert negative log likelihood values to likelihood values.

        _error_convert_f = lambda nll: np.exp(-nll) 

        if method == 'sgd':

            performance_trace = optimizer.sgd(self._neg_log_likelihood_f,
                                            self._gradient_f,
                                            fargs=[X, T_ind_vars], n_epochs=n_epochs,
                                            learning_rate=learning_rate,
                                            error_convert_f=_error_convert_f,
                                            error_convert_name='Likelihood',
                                            verbose=verbose)

        elif method == 'adam':

            performance_trace = optimizer.adam(self._neg_log_likelihood_f,
                                               self._gradient_f,
                                               fargs=[X, T_ind_vars], n_epochs=n_epochs,
                                               learning_rate=learning_rate,
                                               error_convert_f=_error_convert_f,
                                               error_convert_name='Likelihood',
                                               verbose=verbose)

        elif method == 'scg':

            performance_trace = optimizer.scg(self._neg_log_likelihood_f,
                                              self._gradient_f,
                                              fargs=[X, T_ind_vars], n_epochs=n_epochs,
                                              error_convert_f=_error_convert_f,
                                              error_convert_name='Likelihood',
                                              verbose=verbose)

        else:
            raise Exception("method must be 'sgd', 'adam', or 'scg'")

        self.total_epochs += len(performance_trace)
        self.performance_trace += performance_trace

        # Return neural network object to allow applying other methods
        # after training, such as:    Y = nnet.train(X, T, 100, 0.01).use(X)

        return self

    def use(self, X):
        """Return the predicted class and probabilities for input samples as rows in X.
        X assumed to not be standardized.

        Parameters
        ----------
        X : two-dimensional numpy array, unstandardized input samples by rows
            number of samples  by  number of input components, unstandardized

        Returns
        -------
        Predicted classes : two-dimensional array of predicted classes for each sample
            number of samples by 1  of ints or strings, depending on how target classes were specified
        Class probabilities : two_dimensional array of probabilities of each class for each sample
            number of samples by number of outputs (number of classes)
        """

        # Standardize X
        #X = (X - self.X_means) / self.X_stds
        Y = self._forward(X)
        Y_softmax = self._softmax(Y[-1]) 
        classes = np.argmax(Y_softmax, axis=1).reshape(-1, 1)

        return classes, Y


import pandas

# See https://coderzcolumn.com/tutorials/python/simple-guide-to-style-display-of-pandas-dataframes
def confusion_matrix(Y_classes, T, background_cmap=None):
    class_names = np.unique(T)
    table = []
    for true_class in class_names:
        row = []
        for Y_class in class_names:
            row.append(100 * np.mean(Y_classes[T == true_class] == Y_class))
        table.append(row)
    conf_matrix = pandas.DataFrame(table, index=class_names, columns=class_names)
    print('Percent Correct (Actual class in rows, Predicted class in columns')
    if background_cmap:
        return conf_matrix.style.background_gradient(background_cmap='Blues').format('{:.1f}')
    else:
        return conf_matrix

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    plt.ion()

    # Classic XOR problem, first as a regression problem, then as a classfication problem

    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    T = np.array([[0], [1], [1], [0]])

    nnet_reg = NeuralNetwork(2, [10], 1)
    nnet_class = NeuralNetworkClassifier(2, [10], 2)

    nnet_reg.train(X, T, n_epochs=100, method='scg')
    nnet_class.train(X, T, n_epochs=100, method='scg')

    Y_reg = nnet_reg.use(X)
    Y_class, Y_prob = nnet_class.use(X)

    plt.figure(1)
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.plot(nnet_reg.get_performance_trace())
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')

    plt.subplot(2, 2, 2)
    plt.plot(T, 'o')
    plt.plot(Y_reg)
    plt.xticks(range(4), ['0,0', '0,1', '1,0', '1,1'])
    plt.legend(['T', 'Y'])

    plt.subplot(2, 2, 3)
    plt.plot(nnet_class.get_performance_trace())
    plt.xlabel('Epoch')
    plt.ylabel('Data Likelihood')

    plt.subplot(2, 2, 4)
    plt.plot(T + 0.05, 'o')
    plt.plot(Y_class + 0.02, 'o')
    plt.plot(Y_prob)
    plt.xticks(range(4), ['0,0', '0,1', '1,0', '1,1'])
    plt.legend(['T + 0.05', 'Y_class + 0.02', '$P(C=0|x)$', '$P(C=1|x)$'])

    print(confusion_matrix(Y_class, T))
