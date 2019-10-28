import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from implementations import *

def histogram(index, data, columns):
    """Plots the histogram of the feature indexed by index"""
    plt.figure(figsize=(10, 5))
    plt.title("Histogram for {}".format(columns[index]))
    ax = sns.distplot(data[:,index], rug=True)

def log_histogram(index, data, columns):
    """Plots the histogram of the feature indexed by index next to the logarithm transformation of the same feature"""
    f, axes = plt.subplots(1, 2)
    plt.title("Histogram for {}".format(columns[index]))
    minimum = np.nanmin(data[:,index])
    log_data = np.log(1 + data[:,index] - minimum)
    sns.distplot(data[:,index], rug=True, ax=axes[0])
    sns.distplot(log_data, rug=True, ax=axes[1])
    
def pie_chart(y):
    """Plots the piechart of labels y"""
    count= sum([x>0 for x in y])
    labels = '1: Signal', '-1: Background'
    sizes = [count, len(y)-count]
    explode = (0, 0.1)
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%')
    plt.title('Pie chart of the labels in our dataset')
    plt.show()
    
def downsample(y,X,keys):
    """Downsaples data points X to have a balanced labeled dataset"""
    idx_pos = np.argwhere(y==1)
    y_pos, X_pos, keys_pos = y[idx_pos], X[idx_pos], keys[idx_pos]
    idx_neg = np.argwhere(y==-1)
    y_neg, X_neg, keys_neg = y[idx_neg], X[idx_neg], keys[idx_neg]

    nbr_pos = len(y_pos)
    nbr_neg = len(y_neg)
    
    idx = np.random.randint(0, nbr_neg, size=nbr_pos)
    y_neg, X_neg, keys_neg = y_neg[idx], X_neg[idx], keys_neg[idx]
    
    y = np.squeeze(np.concatenate((y_neg, y_pos), axis=0))
    X = np.squeeze(np.concatenate((X_neg, X_pos), axis=0))
    keys = np.squeeze(np.concatenate((keys_neg, keys_pos), axis=0))
    return y,X,keys

def poly_expansion(x, degree):
    """polynomial basis functions for input data x, for j=1 up to j=degree."""
    rows = x.shape[0]
    cols = x.shape[1]*degree
    poly_array = np.zeros((rows,cols))
    for i in range(x.shape[1]):
        poly = np.vander(x[:,i],degree+1,increasing=True)[:,1:]
        for j in range(degree):
            poly_array[:,i*degree+j] = poly[:,j]
    return poly_array

def stack_cols(x,y):
    """Stack columns from y to columns from x"""
    rows = x.shape[0]
    cols = x.shape[1]+y.shape[1]
    stacked_array = np.zeros((rows,cols))
    for i in range(x.shape[1]):
        stacked_array[:,i] = x[:,i]
    for i in range(y.shape[1]):
        stacked_array[:,x.shape[1]+i] = y[:,i]
    return stacked_array

def stack_log(X, idx):
    """Apply log on right skewed features indexed by idx and add them to the original features"""
    rows = X.shape[0]
    cols = len(idx)
    data_with_log = np.zeros((rows,cols))
    for i,j in enumerate(idx):
        minimum = np.nanmin(X[:,j])
        data_with_log[:,i] = np.log(1 + X[:,j] - minimum)
    return data_with_log

def correlation_matrix(data, col_names):
    """Plot the correlation matrix of data"""
    nbr_cols= data.shape[1]
    plt.figure(figsize=(12, 6))
    corr = np.zeros((nbr_cols,nbr_cols))
    corr = np.corrcoef(data.T)        
    ax = sns.heatmap(corr, xticklabels=col_names, yticklabels=col_names, 
                     linewidths=.2, cmap="YlGnBu")
    
def stack_log(X, idx):
    """Apply log on right skewed features indexed by idx and add them to the original features"""
    rows = X.shape[0]
    cols = X.shape[1]+len(idx)
    data_with_log = np.zeros((rows,cols))
    for j in range(X.shape[1]):
        data_with_log[:,j] = X[:,j]
    for i,j in enumerate(idx):
        minimum = np.nanmin(X[:,j])
        data_with_log[:,X.shape[1]+i] = np.log(1 + X[:,j] - minimum)
    return data_with_log

def process_data(y, input_data, keys):
    """Processing pipeline used for training_set that we will also apply for test_set"""
    #1st partition consisting of elements having PRI_jet_num=0
    ids_1 = np.argwhere(input_data[:,22]==0)[:,0]
    keys_1, y_1, X_1 = keys[ids_1], y[ids_1], input_data[ids_1]
    #2nd partition consisting of elements having PRI_jet_num=1
    ids_2 = np.argwhere(input_data[:,22]==1)[:,0]
    keys_2, y_2, X_2 = keys[ids_2], y[ids_2], input_data[ids_2]
    #3rd partition consisting of elements having PRI_jet_num>1
    ids_3 = np.argwhere(input_data[:,22]>1)[:,0]
    keys_3, y_3, X_3 = keys[ids_3], y[ids_3], input_data[ids_3]
    
    #We drop the features that are constant in each subset.
    cols_1 = [4,5,6,12,22,23,24,25,26,27,28,29]
    cols_2 = [4,5,6,12,22,26,27,28]
    cols_3 = []
    X_1 = np.delete(X_1, cols_1, axis=1)
    X_2 = np.delete(X_2, cols_2, axis=1)
    X_3 = np.delete(X_3, cols_3, axis=1)
    
    #We need to handle missing values
    undefined_value = -999.0
    X_1 = replace_nan(X_1, undefined_value, mean=False)
    X_2 = replace_nan(X_2, undefined_value, mean=False)
    X_3 = replace_nan(X_3, undefined_value, mean=False)
    
    #We take the polynomial expansion of our original features
    poly_1 = poly_expansion(X_1, 7)
    poly_2 = poly_expansion(X_2, 7)
    poly_3 = poly_expansion(X_3, 7)
    
    #We take the log value of the features
    idx_1 = range(18)
    X_1_logged = stack_log(X_1, idx_1)
    idx_2 = range(22)
    X_2_logged = stack_log(X_2, idx_2)
    idx_3 = range(30)
    X_3_logged = stack_log(X_3, idx_3)
    
    #We stack the polynomials along with the logarithms
    X_1_stack = stack_cols(poly_1,X_1_logged)
    X_2_stack = stack_cols(poly_2,X_2_logged)
    X_3_stack = stack_cols(poly_3,X_3_logged)
    
    #We standardize the data
    X_1_std = standardize(X_1_stack)
    X_2_std = standardize(X_2_stack)
    X_3_std = standardize(X_3_stack)
    
    return (keys_1, X_1_std, y_1),(keys_2, X_2_std, y_2),(keys_3, X_3_std, y_3)

def replace_nan(input_data, undefined_value, mean=True):
    """replace all entries = undefined_value with mean (or median if mean=False) """
    input_data[input_data == -999] = np.nan
    #the use of np.where provides us with the indices of the nan values
    indices_nan = np.where(np.isnan(input_data))
    if(mean):
        #means is a vector where each element corresponds to the mean of a feature from the input
        means=np.nanmean(input_data, axis=0, keepdims = True)
        #we replace the -999 values with the means of the respective columns
        input_data[indices_nan] = np.take(means, indices_nan[1])
    else:
        medians = np.nanmedian(input_data, axis=0, keepdims = True)
        input_data[indices_nan] = np.take(medians, indices_nan[1])
    return input_data


def standardize(cleaned_data):
    """standardize the already cleaned dataset """
    means=np.mean(cleaned_data, axis=0, keepdims=True)
    #let's compute the data - mean
    data_sub_mean= cleaned_data - means
    #the formula to standardize data is : (data-mean)/std
    #we need to compute the std for the data ignoring the undefined values
    std=np.std(cleaned_data, axis=0, keepdims = True)
    standard_data = data_sub_mean/std
    return standard_data


def success_rate(predicted_labels,true_labels):
    """calculate the success rate of our predictions """
    success_rate = 1 - (np.count_nonzero(predicted_labels - true_labels)/len(predicted_labels))
    return success_rate


def predictions_linear(input_data,weights,threshold):
    """calculate the linear predictions and make them categorical prediction according to a threshold """
    prediction = input_data@weights
    labels_predicted = [1 if x > threshold else -1 for x in prediction]
    return labels_predicted

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

            

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # polynomial basis function: 
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    return np.polynomial.polynomial.polyvander(x, degree)



def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # split the data based on the given ratio:
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    split = int( ratio * len(y) )
    train_indices, test_indices= np.split(indices, np.array([split]))
    train_data = x[train_indices]
    train_labels = y[train_indices]
    test_data = x[test_indices]
    test_labels = y[test_indices]
    return train_data, train_labels, test_data, test_labels




def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def find_threshold(X, y, weights, logistic = False, print_result=False):
    """
    for a given set of observations X and their labels y, finds 
    the optimal threshold that decides whether a prediction value should
    count as a label 1 or a label -1
    """
    x_train, y_train, x_validation, y_validation=split_data(X, y, ratio=0.8)
    prediction_test=x_validation@weights
    thresholds = np.linspace(-0.5, 0.5, 1000)
    max_frac = -999
    for j, i in enumerate(thresholds):
        predicted_labels = [1 if x > i else -1 for x in prediction_test]
        fraction = 1 - (np.count_nonzero(predicted_labels - y_validation)/len(predicted_labels))
        if max_frac < fraction :
            max_frac = fraction 
            max_thresh = thresholds[j]
    if print_result:
        print('best threshold = ',max_thresh, ' with accuracy = ', max_frac)
    return max_thresh


def cross_validation_single_step(y, x, k_indices, k,\
                                 lambda_, degree=False,logistic=False, gamma = 0.1, threshold_provided=False, threshold=0):
    """Takes one step of cross validation for ridge or penalized logistic regression
       """
    
    # determine which data is for training and which data is for testing: 
    x_te,y_te = x[k_indices[k]], y[k_indices[k]]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    x_tr,y_tr = x[tr_indice],y[tr_indice]
    
    #if degree:
        # form data with polynomial degree:
        #x_te = build_poly(x_te,degree)
        #x_tr = build_poly(x_tr,degree)
        
    # run ridge regression to determine the weights:
    if (logistic):
        weights, _ = reg_logistic_regression(y_tr,x_tr, initial_w = np.ones(x_tr.shape[1]), max_iters=1000,lambda_ = lambda_, gamma=gamma)
        #prediction = sigmoid(x_te@weights)
    else:
        weights, _ = ridge_regression(y_tr,x_tr,lambda_)
        #prediction = x_te@weights
    prediction = x_te@weights
    if (threshold_provided):
        thresh = threshold
    else:
        thresh= find_threshold(x_tr, y_tr, weights, print_result=False)
    predicted_labels = [1 if x > thresh else -1 for x in prediction]
    predicted_fraction = 1 - (np.count_nonzero(predicted_labels - y_te)/len(predicted_labels))
    
    # calculate the loss for train and test data:
    loss_tr = np.sqrt(2*compute_loss(y_tr, x_tr,weights))
    loss_te = np.sqrt(2*compute_loss(y_te, x_te,weights))
    
    return loss_tr, loss_te, weights, predicted_fraction


def cross_validate(y, x, k_fold, lambda_range, degree=False, logistic=False, gamma=0.1, threshold_provided=False, threshold=0):
    """Cross validates ridge or penalized logistic regression
        Returns for each lambda
        - the training and testing errors
        - the successfully predicted fraction of labels"""
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold)
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    predicted_fractions = []
    
    # for each lambda, compute the test error, the train error and the well predicted fraction of labels
    for lambda_ in lambda_range:
        rmse_tr_inner = []
        rmse_te_inner = []
        predicted_fraction_inner = []
        
        #create inner arrays of errors and predicted fractions of size k, of which we will take the average
        for k in range(k_fold):
            loss_tr, loss_te, weights, predicted_fraction =  cross_validation_single_step(y,\
                                                                                          x,k_indices, k, lambda_, degree, logistic,\
                                                                                          gamma, threshold_provided, threshold)
            rmse_tr_inner.append(loss_tr)
            rmse_te_inner.append(loss_te)
            predicted_fraction_inner.append(predicted_fraction)
            
        # Average the k results
        rmse_tr.append(np.mean(rmse_tr_inner))
        rmse_te.append(np.mean(rmse_te_inner))
        predicted_fractions.append(np.mean(predicted_fraction_inner))
    
    return rmse_tr, rmse_te, predicted_fractions



def indices_correlated_features(corr_matrix, corr_threshold, col_names):
    """ 
    Arguments: correlation matrix, the names of the features and a threshold of correlation
    Returns the pairs of correlated features
    """
    indices=[]
    names=[]
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            if abs(corr_matrix[i][j]) >= corr_threshold:
                indices.append((i,j))
                names.append((col_names[i],col_names[j]))
    return indices, names
