"""
This is an implementation of a few kernels for the challenge. The implementation is based on the coding homework from the class.
"""

import numpy as np

class RBF:
    def __init__(self, sigma=1.):
        self.sigma = sigma  ## the variance of the kernel
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        X2,Y2 = np.linalg.norm(X, ord=2, axis=1, keepdims=True)**2, np.linalg.norm(Y, ord=2, axis=1, keepdims=True)**2
        XY = X@Y.T
        aux = X2+Y2.T-2*XY
        return np.exp(-aux/(2*self.sigma**2)) ## Matrix of shape NxM
    
class Linear:
    #def __init__(self):
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        return X@Y.T## Matrix of shape NxM

class Polynomial:
    def __init__(self, degree=4):
        self.degree = degree  ## the variance of the kernel
    def kernel(self,X,Y):
        return (X@Y.T)**self.degree ## Matrix of shape NxM