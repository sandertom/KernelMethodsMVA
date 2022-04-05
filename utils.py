import numpy as np
from tqdm import tqdm
import numpy as np
import pandas as pd
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from utils import *
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.svm import SVC
from SVM import OVO_test, OVO_train
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def rgb2gray(x):
    #last axis of x must be the number of axis
    return x@[0.2125, 0.7154, 0.0721]



def extract_random_patch(image, rf_size):
    corner = np.random.randint(32 - rf_size, size=2)
    i,j = corner[0], corner[1]
    patch = image[i:i+rf_size, j:j+rf_size, :]
    patch = patch - np.mean(patch)
    patch/=(np.std(patch)+1)
    return patch

def extract_patches(dataset, num_patches, rf_size):
    print("Extracting patches...")
    patches = np.zeros((num_patches, rf_size*rf_size*3))
    for i in tqdm(range(num_patches)):
        img_idx = i%len(dataset)
        patch = extract_random_patch(dataset[img_idx], rf_size)
        patches[i] = patch.reshape(-1)
    return patches 

def cluster_patches(patches, k):
    print("Clustering patches...")
    clust = KMeans(n_clusters = k, max_iter = 50, n_init = 5, verbose= True)
    clust.fit(patches)
    return clust.cluster_centers_, clust.labels_

def whiten_patches(patches):
    
    C = np.cov(patches, rowvar=False)  
    M = np.mean(patches, axis=0)
    d, V = np.linalg.eigh(C)
    D = np.diag(np.sqrt(1. / (d + 0.1)))
    P = np.matmul(np.matmul(V, D), V.T)
    feats = np.matmul(patches - M, P)

    return feats, M, P 

def get_quarters(img):
    img_size = img.shape[0]
    half_size = img_size//2
    q1 = img[:half_size, :half_size,:]
    q2 = img[half_size:, :half_size,:]
    q3 = img[:half_size, half_size:,:]
    q4 = img[half_size:, half_size:,:]
    return [q1, q2, q3, q4]

def compute_features_patch(patch, centroids):
    k = len(centroids)
    z = np.zeros(k)
    for i, centroid in enumerate(centroids):
        z[i] = np.linalg.norm(patch - centroid)
    f =  np.maximum(np.zeros(k), np.mean(z)*np.ones(k) - z)
    return f

def compute_features_img(img, centroids, stride, patch_size, M, P):
    k = len(centroids)
    img_feat = np.zeros(k)
    img_size = img.shape[0]
    i = 0
    while i+patch_size < img_size:
        j = 0
        while j+patch_size < img_size:
            patch = img[i:i+patch_size, j:j+patch_size,:]
            patch = patch.reshape(-1)
            
            patch = (patch - patch.mean())/(patch.std()+1) #normalize
            patch = np.matmul(patch - M, P) #whiten
            j+=stride
            f = compute_features_patch(patch, centroids)
            img_feat += f
            #print(i,j)
        i+=stride
    return img_feat

def compute_features(dataset, centroids, stride, patch_size, M, P):

    X = []    
    for idx in tqdm(range(len(dataset))):
        img = dataset[idx]
        quart = get_quarters(img)
        f = []
        for i in range(4):
            quarter_feats = compute_features_img(quart[i], centroids, stride, patch_size, M, P)
            f.append(quarter_feats) #k features per quarter
        #print(f.shape)
        X.append(f)
    X = np.array(X)
    X = X.reshape(len(X), -1)
    return X

def post_processing(X):
    X -= X.mean(axis=0, keepdims=True)
    X /= (0.01 + np.std(X, axis=0))
    #X = np.hstack([X, np.ones((len(X), 1))])
    return X


