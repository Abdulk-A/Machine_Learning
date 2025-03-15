import numpy as np
class RidgeRegression:
    
    def __init__(self, lam=1.0):
        self.lam = lam
        self.beta_coeffs = None
        
    #addings 1's the first value of each row to account for the bias term 
    def bias_column(self, x):
        return np.insert(x,0,1,axis=1)
    
    def predict(self, X):
        #making sure user has use the fit method
        if self.beta_coeffs is None:
            raise ValueError('Model has not been trained.')

        x_train = self.bias_column(X)
        return np.dot(x_train, self.beta_coeffs) 

    #x: features ndarray, y: label ndarray
    #formula: ( (x_tranpose * x) + (lambda * Identity_matrix) )^(-1) * x_tranpose * y
    def fit(self, X, y): 
        
        #add the bias term to matrix   
        X = self.bias_column(X)
        n, p = X.shape

        x_t = np.transpose(X)
        
        identity_matrix = np.identity(p)
        identity_matrix[0][0] = 0

        self.beta_coeffs = np.dot(np.dot(np.linalg.inv(np.dot(x_t, X) + (self.lam * identity_matrix)), x_t), y)

