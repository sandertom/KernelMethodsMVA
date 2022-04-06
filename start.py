import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from SVM import OVO_test, OVO_train
import kernels
import argparse
from hog import hog
from utils import decision

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--combine", default=False, help="Wether to combine both methods or not")
args = parser.parse_args()
config = vars(args)

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
print("extracting patches..")
patches = extract_patches(X_tot, 200000, 6)

##Whiten
print("whitening patches...")
patches, M, P = whiten_patches(patches)

##Cluster
print("computing centroids...")
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
_, pred = OVO_test(X_test, dic)

if config["combine"]:
    print("training the hog model!")
    print("computing hog features...")
    Xtr_hog = np.array([hog(image) for image in Xtr])
    Xte_hog = np.array([hog(image) for image in Xte])
    print("Fitting SVM...")
    dic2 = OVO_train(Xtr_hog, Ytr, sigma = 3, C = 1) #default is Laplacian kernel
    allvotes,_ = OVO_test(Xte_hog, dic2)
    pred = np.array([decision(pred[i],allvotes[i]) for i in range(len(pred))])

print("done!")
aux = pd.DataFrame(pred).reset_index()
aux['index']+=1
aux.rename(columns={'index':'Id', 0:"Prediction"}, inplace=True)
pd.DataFrame(aux).to_csv("pred", index=False)