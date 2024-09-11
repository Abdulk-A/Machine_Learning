import numpy as np

#addings 1's the first value of each row to account for the bias term 
def bias_column(x):
    return np.insert(x,0,1,axis=1)

#x,y input matrices , lam is the one of the 5 manually chosen values, n is num of rows, p is cols
#returns beta coeffs
def ridge_regression_fit(x, y, lam): 
    #formula: ( (x_tranpose * x) + (lambda * Identity_matrix) )^(-1) * x_tranpose * y

    #add the bias term to matrix   
    x = bias_column(x)
    n, p = x.shape

    x_t = np.transpose(x)
    
    identity_matrix = np.identity(p)
    identity_matrix[0][0] = 0

    return np.dot(np.dot(np.linalg.inv(np.dot(x_t, x) + (lam * identity_matrix)), x_t), y)

def ridge_regression_predict(beta_coeffs, x_train):
    x_train = bias_column(x_train)
    return np.dot(x_train, beta_coeffs) 