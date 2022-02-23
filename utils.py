def rgb2gray(x):
    #last axis of x must be the number of axis
    return x@[0.2125, 0.7154, 0.0721]