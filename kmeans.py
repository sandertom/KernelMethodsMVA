import numpy as np

class Kmeans:
    """
    m: number of clusters (assumed between 0 and m - 1)
    d: dimension of the space in which the data lives
    assignments: assignment
    mu: centers of different clusters
    """
    def __init__(self, m):
        self.m = m #number of clusters
        self.dim = None
        self.assignments = None
        self.mu = None

    def assign(self, data):
        #assign each point to the closest center
        n = len(data)
        for i in range(n):
            dmin = np.linalg.norm(data[i, :] - self.mu[0, :])
            for k in range(1, self.m):
                dist = np.linalg.norm(data[i, :] - self.mu[k, :])
                if dist < dmin:
                    dmin = dist
                    self.assignments[i] = k

    def update(self, data):
        #uptade each centroid
        n = len(data)
        aux = np.zeros(self.m) #counts the number of points assigned to each cluster
        self.mu = np.zeros((self.m, self.dim))
        for i in range(n):
            aux[self.assignments[i]] += 1
            self.mu[self.assignments[i], :] += data[i, :]
        for k in range(self.m):
            if aux[k] > 0:
                self.mu[k, :] /= aux[k]

    def fit(self, data, n_iter=20):
        #fitting kmeans with 20 iterations
        self.dim = len(data[0])
        self.assignments = np.zeros(data.shape[0], dtype=int)
        index = np.random.random_integers(0, data.shape[0] - 1, (self.m)) #choose random centroids at start from the data
        self.mu = np.zeros((self.m, self.dim))
        for i, idx in enumerate(index):
            self.mu[i] = data[idx, :]

        for i in range(n_iter):
            self.assign(data)
            self.update(data)

    def predict(self, X):
        #assigns each point to its cluster
        aux = np.zeros(self.m)
        y = np.zeros(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            dmin = np.linalg.norm(x - self.mu[0, :])
            for k in range(1, self.m):
                dist = np.linalg.norm(x - self.mu[k, :])
                if dist < dmin:
                    y[i] = k
                    dmin = dist
            aux[y[i]] += 1
        return y