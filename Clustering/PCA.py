import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
import matplotlib.cm as cm

digits = load_digits()
XDigits, YDigits = digits.data, digits.target

images_and_labels = list(zip(digits.images, digits.target))

for index, (image,label) in enumerate(images_and_labels[:6]):
    plt.subplot(2,3, index+1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Target: %i' % label)

plt.show()

estimator = PCA(n_components=10)
X_pca = estimator.fit_transform(XDigits)

colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']

for i in range(len(colors)):
    px = X_pca[:,0][YDigits == i]
    py = X_pca[:,1][YDigits == i]
    plt.scatter(px,py, c=colors[i])
    plt.legend(digits.target_names)
    plt.xlabel('First Principle Component')
    plt.ylabel('Second Principle Component')

plt.show()
