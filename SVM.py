import numpy as np
import pickle as pkl
from scipy import optimize

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

class KernelSVC:
    
    def __init__(self, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
       
    
    def fit(self, X, y):
       #### You might define here any variable needed for the rest of the code
        N = len(y)
        K = self.kernel(X,X)
        y_diag = np.diag(y)

        # Lagrange dual problem
        def loss(alpha):
            return  0.5*alpha.T@y_diag@K@y_diag@alpha - np.sum(alpha)#'''--------------dual loss ------------------ '''

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            return y_diag@K@y_diag@alpha - np.ones(N)# '''----------------partial derivative of the dual loss wrt alpha-----------------'''


        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0

        fun_eq = lambda alpha:  (-y.T@alpha).reshape((1,1))# '''----------------function defining the equality constraint------------------'''        
        jac_eq = lambda alpha:   -y.reshape((1,N)) #'''----------------jacobian wrt alpha of the  equality constraint------------------'''
        fun_ineq = lambda alpha:  self.C*np.vstack((np.ones((N,1)),np.zeros((N,1))))-(np.vstack((np.eye(N),-np.eye(N)))@alpha).reshape((2*N,1)) # '''---------------function defining the ineequality constraint-------------------'''     
        jac_ineq = lambda alpha:  -np.vstack((np.eye(N),-np.eye(N))) # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''
        
        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq', 
                        'fun': fun_ineq , 
                        'jac': jac_ineq})

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N), 
                                   method='SLSQP', 
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints)
        self.alpha = optRes.x

        ## Assign the required attributes
        
        supportIndices = np.logical_and(self.alpha-self.epsilon>0,self.alpha+self.epsilon<self.C)
        self.support = X[supportIndices] #'''------------------- A matrix with each row corresponding to a support vector ------------------'''
        self.b = (y[supportIndices] - (K@y_diag@self.alpha)[supportIndices]).mean()#''' -----------------offset of the classifier------------------ '''
        self.norm_f = self.alpha.T@y_diag@K@y_diag@self.alpha# '''------------------------RKHS norm of the function f ------------------------------'''
        self.support2 = X[self.alpha-self.epsilon>0]
        self.alpha2 = (y_diag@self.alpha)[self.alpha-self.epsilon>0]

    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return self.kernel(x, self.support2)@self.alpha2
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b> 0) - 1

def OVO_train(X_train, y_train, sigma, C):
    dic = {}
    for i in range(9):
        print("value of i: {}".format(i))
        for j in range(i+1,10):
            print("value of j: {}".format(j))
            indicies1 = y_train==i
            indicies2 = y_train==j
            y_train_ = y_train.copy()
            y_train_[indicies1] = 1
            y_train_[indicies2] = -1
            y_train_ = y_train_[np.logical_or(indicies1, indicies2)]
            X_train_ = X_train[np.logical_or(indicies1, indicies2)]
            kernel = RBF(sigma).kernel
            model = KernelSVC(C=C, kernel=kernel)
            model.fit(X_train_, y_train_)
            dic[(i,j)]=model
    return dic

def OVO_test(X_test, dic):
    res = [[0 for i in range(10)] for j in range(len(X_test))]
    for k in range(len(X_test)):
        if k%50==0:
            print(k)
        for i in range(9):
            for j in range(i+1, 10):
                if dic[(i,j)].predict([X_test[k]])==1:
                    res[k][i]+=1
                else:
                    res[k][j]+=1
    res = np.argmax(res, axis=1)
    return res