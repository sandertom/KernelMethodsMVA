This is a test

Liens intéressants:

"Faces recognition example using eigenfaces and SVMs"
https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py
-> Semble pas fonctionner pour nous je comprends pas pourquoi 

"Image denoising using kernel PCA"
https://scikit-learn.org/stable/auto_examples/applications/plot_digits_denoising.html#sphx-glr-auto-examples-applications-plot-digits-denoising-py
-> J'ai testé PCA, KernelPCA, faire des inverse transforme puis hog pis svm, ça améliore pas les nerfs.... Voire notebook

"Plot classification probability"
https://scikit-learn.org/stable/auto_examples/classification/plot_classification_probability.html#sphx-glr-auto-examples-classification-plot-classification-probability-py
Il faudrait tester le OVR, pour ça il faut que la SVC renvoie des probes d'appartenance...

"Online learning of a dictionary of parts of faces"
https://scikit-learn.org/stable/auto_examples/cluster/plot_dict_face_patches.html#sphx-glr-auto-examples-cluster-plot-dict-face-patches-py
Dictionary learning, maybe next step ?

"K-means clustering"
https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
Maybe we can look what K-means does ?

Image denoising using dictionary learning
https://scikit-learn.org/stable/auto_examples/decomposition/plot_image_denoising.html#sphx-glr-auto-examples-decomposition-plot-image-denoising-py
-> rien compris mais ça peut être intéressant

https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html#sphx-glr-auto-examples-decomposition-plot-faces-decomposition-py
-< pareil

CMU with patches https://trucvietle.me/files/601-report.pdf
Article original de NG bcp mieux expliqué: http://proceedings.mlr.press/v15/coates11a/coates11a.pdf
Une implémentation from scratch récente : https://github.com/alisher-ai/unsupervised-feature-learning