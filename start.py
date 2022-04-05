import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from SVM import OVO_test, OVO_train
import kernels

from kmeans import kmeans_centroids
from utils import extract_patches, whiten_patches, compute_features, post_processing

Xtr = pd.read_csv('data/Xtr.csv',header=None,sep=',',usecols=range(3072))
Xte = pd.read_csv('data/Xte.csv',header=None,sep=',',usecols=range(3072))
Ytr_ = pd.read_csv('data/Ytr.csv')#, index_col = None, header = None)

Xtr = np.array(Xtr).reshape(5000, 3,32, 32).swapaxes(1,2).swapaxes(2,3)
Xte = np.array(Xte).reshape(2000, 3,32, 32).swapaxes(1,2).swapaxes(2,3)
Ytr = np.array(Ytr_['Prediction'])
X_tot = np.vstack([Xtr, Xte]) #Entire dataset used for feature extraction

##Extract
patches = extract_patches(X_tot, 200000, 6)
##Whiten
patches, M, P = whiten_patches(patches)
##Cluster
centroids = kmeans_centroids(patches, num_centroids = 400, n_iter = 50)

##Compute features
print("Computing train features...")
X_train = compute_features(Xtr, centroids, stride=10, patch_size = 6, M = M, P = P)
print("Computing test features...")
X_test = compute_features(Xte, centroids, stride=10, patch_size = 6, M = M, P = P)

##Scale 
X_train = post_processing(X_train)
X_test = post_processing(X_test)

print("Fitting SVM...")
dic = OVO_train(X_train, Ytr, sigma = 60, C = 10, ker = kernels.LaplacianRBFKernel)
pred = OVO_test(X_test, dic)
np.save('pred_script.npy', pred_script)