As part of the course "Kernel Methods in Machine Learning" (Julien Mairal and Michaël Arbel, MVA, 2022), this assignment aims to implement kernel methods for image classification. We approach the problem in a particular context: no Deep Learning, use of kernel methods, and implementation of algorithms from scratch. This repository contains our work. 

We have implemented two different methods, that give approximatively the same accuracy. Combining them helped us to improve slightly the accuracy.

- First method: Laplacian kernel SVM on HOG features
- Second method: whitenting + Unsupervised feature extraction using kmeans + Gaussian SVM inspired from http://proceedings.mlr.press/v15/coates11a/coates11a.pdf
- Third method: looking at the votes of each class in the one vs one SVM of the first method. Choosing among the best candidates using the secong method.

These results can be reproduced by running:
```
python start.py
```
By default, only the first method is used. To combine both methods as previously explained, please use:
```
python start.py --combine True
```
In the other notebooks, you can find other things we have tried 
- Data augmentation (did not help)
- Kernel PCA (did not help)
- Denoising (did not help)
- Histogram of colors features (did not help)
- Using other Kernels

Please note that since it is our implementation of SVM, it is slow. It is from far the bottleneck in terms of complexity. So if you use the "--combine True" command line, it will be twice slower, even though the performances are (slightly) better. But don't worry, it converges! we don't know why it takes that much time compared to sklearn. Maybe a problem with the support vectors in the dual formulation? or in our One VS One implementation for multiclass classification? issues are welcomed!

Thanks!
Best regards,
Jean and Tom
