import numpy as np
from helpers import *


def compute_gradient(y, tx, w):
    """Computes the gradient"""
    error = y - tx@w
    return -(tx.T@error)/(y.shape[0])

def compute_loss(y, tx, w):
    """Computes the loss by MSE"""
    error = y - tx@w
    return 1/2 * np.mean(error**2)


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Performs the gradient descents algorithm and returns the last weights and loss value"""
    
    # initialize the weights
    w = initial_w
    
    for n_iter in range(max_iters):
        
        # compute loss and gradient
        loss=compute_loss(y, tx, w)
        gradient=compute_gradient(y, tx, w)
        
        # update the weights 
        w = w - gamma * gradient
        
        if n_iter % 200 == 0:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return w, loss




def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Performs the stochastic gradient descents algorithm and returns the last weights and loss value"""
    
    # initialise weights
    w = initial_w
    
    for n_iter in range(max_iters):
        
        # only take into account data points from the mini-batch
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size = 100):
            
            # compute loss and gradient
            loss=compute_loss(y, tx, w)
            gradient=compute_gradient(minibatch_y, minibatch_tx, w)
            
            # update weights
            w = w - gamma * gradient
        if n_iter % 200 == 0:
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}".format(
                      bi=n_iter, ti=max_iters - 1, l=loss))
    
    return w, loss


def least_squares(y, tx):
    """Computes the least squares solution using normal equations"""
    # the linear system we need to solve is: (X.T@X)w* = X.T@y
    left=tx.T@tx
    right=tx.T@y
    w,_,_,_ = np.linalg.lstsq(left,right)
    
    # compute loss
    loss = compute_loss(y,tx,w)
    
    return w, loss



def ridge_regression(y, tx, lambda_):
    """Implements ridge regression"""
    #define lambda_prime
    lambda_pr=lambda_*2*len(y)
    
    #compute weight
    left=(tx.T@tx)+ (lambda_pr*np.identity(tx.shape[1]))
    right=(tx.T@y)
    w,_,_,_ = np.linalg.lstsq(left,right)
    
    #compute the loss
    loss=compute_loss(y, tx, w)
    
    return w, loss


################################
     # LOGISTIC REGRESSION #
###############################

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))
    

    
def calculate_loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood."""
    inner = np.dot(tx,w)
    eps = 1e-10
    sig = sigmoid(inner)
    sig[sig < eps] = eps
    sig_1 = 1 - sig
    sig_1[sig_1<eps] = eps
    loss = (-1) * (np.dot(y.T,np.log(sig)) + np.dot((1-y).T,np.log(sig_1)))
    return np.squeeze(loss)
 
    
    
def calculate_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    prediction = sigmoid(tx@w)
    return tx.T@(prediction-y) / y.shape[0]



def logistic_regression(y, tx, initial_w, max_iters, gamma):    
    # inital weights
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        
        # get loss and update w.
        loss=calculate_loss_logistic(y,tx,w)
        gradient=calculate_gradient_logistic(y,tx,w)
        
        w=w-gamma*gradient
        
        # log info
        #if iter % 500 == 0:
         #   print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
    return w, loss



def reg_logistic_regression(y, tx,  lambda_, initial_w, max_iters, gamma):    
    #initial weights
    w = initial_w
    
    # start the logistic regression
    for iter in range(max_iters):
        
        # get loss and update w.
        loss =  calculate_loss_logistic(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        gradient = calculate_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        
        w -= gamma * gradient
       
        # log info
        if iter % 500 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
    return w, loss