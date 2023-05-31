
import numpy as np
import copy
import math
import sys  # for sys.float_info.epsilon

######################################################################
## class Optimizers()
######################################################################

class Optimizers():

    def __init__(self, all_weights):
        '''all_weights is a vector of all of a neural networks weights concatenated into a one-dimensional vector'''
        
        self.all_weights = all_weights

        self.sgd_initialized = False
        self.scg_initialized = False
        self.adam_initialized = False

    ######################################################################
    #### sgd
    ######################################################################

    def sgd(self, error_f, gradient_f, fargs=[], n_epochs=100, learning_rate=0.001,
            error_convert_f=None, error_convert_name='RMSE', nesterov=False, callback_f=None, verbose=True):
        '''
        error_f: function that requires X and T as arguments (given in fargs) and returns mean squared error.
        gradient_f: function that requires X and T as arguments (in fargs) and returns gradient of mean squared error
                    with respect to each weight.
        error_convert_f: function that converts the standardized error from error_f to original T units.
        '''

        if not self.sgd_initialized:
            error_trace = []
            self.momentum = 0.9
            self.prev_update = 0
            if nesterov:
                self.all_weights_copy = np.zeros(self.all_weights.shape)
 
        epochs_per_print = n_epochs // 10

        for epoch in range(n_epochs):

            error = error_f(*fargs)
            grad = gradient_f(*fargs)

            if not nesterov:
                
                self.prev_update = learning_rate * grad + self.momentum * self.prev_update
                # Update all weights using -= to modify their values in-place.
                self.all_weights -= self.prev_update

            else:
                self.all_weights_copy[:] = self.all_weights 

                self.all_weights -= self.momentum * self.prev_update
                error = error_f(*fargs)
                grad = gradient_f(*fargs)
                self.prev_update = learning_rate * grad + self.momentum * self.prev_update
                self.all_weights[:] = self.all_weights_copy
                self.all_weights -= self.prev_update
                
            if error_convert_f:
                error = error_convert_f(error)
            error_trace.append(error)

            if callback_f is not None:
                callback_f(epoch)

            if verbose and ((epoch + 1) % max(1, epochs_per_print) == 0):
                print('sgd: Epoch {} {}={:.5f}'.format(epoch + 1, error_convert_name, error))

        return error_trace

    ######################################################################
    #### adam
    ######################################################################

    def adam(self, error_f, gradient_f, fargs=[], n_epochs=100, learning_rate=0.001,
             error_convert_f=None, error_convert_name='RMSE', callback_f=None, verbose=True):
        '''
        error_f: function that requires X and T as arguments (given in fargs) and returns mean squared error.
        gradient_f: function that requires X and T as arguments (in fargs) and returns gradient of mean squared error
                    with respect to each weight.
        error_convert_f: function that converts the standardized error from error_f to original T units.
        '''

        if not self.adam_initialized:
            shape = self.all_weights.shape
            # with multiple subsets (batches) of training data.
            self.mt = np.zeros(shape)
            self.vt = np.zeros(shape)
            self.sqrt = np.sqrt
                
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.beta1t = 1
            self.beta2t = 1
            self.adam_initialized = True

        alpha = learning_rate  # learning rate called alpha in original paper on adam
        epsilon = 1e-8
        error_trace = []
        epochs_per_print = n_epochs // 10

        for epoch in range(n_epochs):

            error = error_f(*fargs)
            grad = gradient_f(*fargs)

            self.mt[:] = self.beta1 * self.mt + (1 - self.beta1) * grad
            self.vt[:] = self.beta2 * self.vt + (1 - self.beta2) * grad * grad
            self.beta1t *= self.beta1
            self.beta2t *= self.beta2

            m_hat = self.mt / (1 - self.beta1t)
            v_hat = self.vt / (1 - self.beta2t)

            # Update all weights using -= to modify their values in-place.
            self.all_weights -= alpha * m_hat / (self.sqrt(v_hat) + epsilon)
    
            if error_convert_f:
                error = error_convert_f(error)
            error_trace.append(error)

            if callback_f is not None:
                callback_f(epoch)

            if verbose and ((epoch + 1) % max(1, epochs_per_print) == 0):
                print('Adam: Epoch {} {}={:.5f}'.format(epoch + 1, error_convert_name, error))

        return error_trace

    ######################################################################
    #### scg
    ######################################################################

    def scg(self, error_f, gradient_f, fargs=[], n_epochs=100,
            error_convert_f=None, error_convert_name='RMSE', callback_f=None, verbose=True):

        if not self.scg_initialized:
            shape = self.all_weights.shape
            self.w_new = np.zeros(shape)
            self.w_temp = np.zeros(shape)
            self.g_new = np.zeros(shape)
            self.g_old = np.zeros(shape)
            self.g_smallstep = np.zeros(shape)
            self.search_dir = np.zeros(shape)
            self.scg_initialized = True

        sigma0 = 1.0e-6
        fold = error_f(*fargs)
        error = fold
        self.g_new[:] = gradient_f(*fargs)
        self.g_old[:] = copy.deepcopy(self.g_new)
        self.search_dir[:] = -self.g_new
        success = True				# Force calculation of directional derivs.
        nsuccess = 0				# nsuccess counts number of successes.
        beta = 1.0e-6				# Initial scale parameter. Lambda in Moeller.
        betamin = 1.0e-15 			# Lower bound on scale.
        betamax = 1.0e20			# Upper bound on scale.
        nvars = len(self.all_weights)
        iteration = 1				# j counts number of iterations
        error_trace = []

        if error_convert_f:
            error = error_convert_f(error)
        error_trace.append(error)

        # Main optimization loop.
        while iteration <= n_epochs:

            # Calculate first and second directional derivatives.
            if success:
                mu = self.search_dir @ self.g_new
                if mu >= 0:
                    self.search_dir[:] = - self.g_new
                    mu = self.search_dir.T @ self.g_new
                kappa = self.search_dir.T @ self.search_dir
                if math.isnan(kappa):
                    print('kappa', kappa)

                if kappa < sys.float_info.epsilon:
                    return error_trace

                sigma = sigma0 / math.sqrt(kappa)

                self.w_temp[:] = self.all_weights
                self.all_weights += sigma * self.search_dir
                error_f(*fargs)  # forward pass through model for intermediate variable values for gradient
                self.g_smallstep[:] = gradient_f(*fargs)
                self.all_weights[:] = self.w_temp

                theta = self.search_dir @ (self.g_smallstep - self.g_new) / sigma
                if math.isnan(theta):
                    print('theta', theta, 'sigma', sigma, 'search_dir[0]', self.search_dir[0], 'g_smallstep[0]', self.g_smallstep[0])

            ## Increase effective curvature and evaluate step size alpha.

            delta = theta + beta * kappa
            # if math.isnan(scalarv(delta)):
            if math.isnan(delta):
                print('delta is NaN', 'theta', theta, 'beta', beta, 'kappa', kappa)
            elif delta <= 0:
                delta = beta * kappa
                beta = beta - theta / kappa

            if delta == 0:
                success = False
                fnow = fold
            else:
                alpha = -mu / delta
                ## Calculate the comparison ratio Delta
                self.w_temp[:] = self.all_weights
                self.all_weights += alpha * self.search_dir
                fnew = error_f(*fargs)
                Delta = 2 * (fnew - fold) / (alpha * mu)
                if not math.isnan(Delta) and Delta  >= 0:
                    success = True
                    nsuccess += 1
                    # w[:] = wnew
                    fnow = fnew

                    if callback_f:
                        callback_f(iteration)

                else:
                    success = False
                    fnow = fold
                    self.all_weights[:] = self.w_temp

            iterationsPerPrint = math.ceil(n_epochs/10)
            if verbose and iteration % max(1, iterationsPerPrint) == 0:
                print('SCG: Iteration {} {}={:.5f}'.format(iteration, error_convert_name, error_convert_f(fnow)))

            if error_convert_f:
                error = error_convert_f(fnow)
            else:
                error = fnow
            error_trace.append(error)

            if success:

                fold = fnew
                self.g_old[:] = self.g_new
                self.g_new[:] = gradient_f(*fargs)

                # If the gradient is zero then we are done.
                gg = self.g_new @ self.g_new  # dot(gradnew, gradnew)
                if gg == 0:
                    return error_trace

            if math.isnan(Delta) or Delta < 0.25:
                beta = min(4.0 * beta, betamax)
            elif Delta > 0.75:
                beta = max(0.5 * beta, betamin)

            # Update search direction using Polak-Ribiere formula, or re-start
            # in direction of negative gradient after nparams steps.
            if nsuccess == nvars:
                self.search_dir[:] = -self.g_new
                nsuccess = 0
            elif success:
                gamma = (self.g_old - self.g_new) @ (self.g_new / mu)
                #self.search_dir[:] = gamma * self.search_dir - self.g_new
                self.search_dir *= gamma
                self.search_dir -= self.g_new

            iteration += 1

            # If we get here, then we haven't terminated in the given number of
            # iterations.

        return error_trace
