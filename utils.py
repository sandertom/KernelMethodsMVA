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



def extract_patch(image, patch_size):
    corner = np.random.randint(32 - patch_size, size=2)
    i,j = corner[0], corner[1]
    patch = image[i:i+patch_size, j:j+patch_size, :]
    patch = patch - np.mean(patch)

    patch/=(np.std(patch)+0.01)
    return patch


def extract_patches(dataset, num_patches, patch_size):
    n_images = len(dataset)
    num_patches_per_image = num_patches//n_images
    patches = np.zeros((n_images, num_patches_per_image, patch_size, patch_size, 3))
    for i in tqdm(range(n_images)):
        for j in range(num_patches_per_image):
            patch = extract_patch(dataset[i], patch_size)
            patches[i, j] = patch
    return patches

def cluster_patches(patches, k):
    n = len(patches)
    data = patches.reshape(n,-1)
    clust = KMeans(n_clusters=k)
    clust.fit(data)
    return clust.labels_, clust.cluster_centers_

def whiten_patches(patches):
    """ whiten """
    feats = patches.reshape(len(patches), -1)
    C = np.cov(feats, rowvar=False)  # 108 x 108 (for 6x6x3 kernels)
    M = np.mean(feats, axis=0)
    d, V = np.linalg.eigh(C)
    D = np.diag(np.sqrt(1. / (d + 0.1)))
    P = np.matmul(np.matmul(V, D), V.T)
    feats = np.matmul(feats - M, P)

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
        z[i] = np.linalg.norm(patch.reshape(-1) - centroid)
    f =  np.maximum(np.zeros(k), np.mean(z)*np.ones(k) - z)
    return f

def compute_features_img(img, centroids, stride, patch_size):
    k = len(centroids)
    img_feat = np.zeros(k)
    img_size = img.shape[0]
    i = 0
    while i+patch_size < img_size:
        j = 0
        while j+patch_size < img_size:
            patch = img[i:i+patch_size, j:j+patch_size,:]
            j+=stride
            f = compute_features_patch(patch, centroids)
            img_feat += f
            #print(i,j)
        i+=stride
    return img_feat

def compute_features(dataset, centroids, stride, patch_size):
    X = []    
    for img in tqdm(dataset):
        quart = get_quarters(img)
        f = []
        for i in range(4):
            quarter_feats = compute_features_img(quart[i], centroids, stride, patch_size)
            #shape num_patches_in_quarter x k
            f.append(quarter_feats) #only k features per quarter
        #print(f.shape)
        X.append(f)
    X = np.array(X)
    X = X.reshape(len(X), -1)
    X -= X.mean(axis=0, keepdims=True)
    #X /= np.std(X, axis=0)
    return X
def plot_image(image, ax):
    im = np.zeros_like(image)
    for i in range(3):
        channel_min = np.min(image[:,:,i], keepdims=True)
        channel_max = np.max(image[:,:,i], keepdims=True)
        im[:,:,i] = (image[:,:,i] - channel_min) / (channel_max - channel_min)
    ax.imshow(im)

