import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

def kmeans_centroids(patches, num_centroids, n_iter):
    
    n_patches = len(patches)
    centroids = 0.1*np.random.randn(num_centroids, patches.shape[1])
    batch = 1000

    for _ in tqdm(range(n_iter)):
        c_squared = (np.power(centroids, 2).sum(1)/2).reshape(-1, 1)
        cluster_sizes = np.zeros((num_centroids, 1))
        Sum = np.zeros((num_centroids, patches.shape[1]))

        for i in np.arange(0, n_patches, batch):
            idx_end = min(i + batch, n_patches)
            bs = idx_end - i

            mat = centroids@patches[i:idx_end, :].T - c_squared
            values, indices = mat.max(0), mat.argmax(0)

            S = csr_matrix((np.ones(bs), (np.arange(bs), indices)), [bs, num_centroids])
            Sum = Sum + S.T * patches[i:idx_end, :]
            cluster_sizes += S.sum(0).T

        centroids = Sum / cluster_sizes
        centroids[np.where(cluster_sizes == 0)[0], :] = 0

    return centroids

import numpy as np
from tqdm import tqdm

class Kmeans:
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

    def fit(self, data, n_iter=50, verbose = True):
        #fitting kmeans with 50 iterations
        self.dim = len(data[0])
        self.assignments = np.zeros(data.shape[0], dtype=int)
        index = np.random.random_integers(0, data.shape[0] - 1, (self.m)) #choose random centroids at start from the data
        self.mu = np.zeros((self.m, self.dim))
        for i, idx in enumerate(index):
            self.mu[i] = data[idx, :]

        for i in tqdm(range(n_iter)):
            if verbose:
                print("Kmeans iteration number {}/{}".format(i+1, n_iter))
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