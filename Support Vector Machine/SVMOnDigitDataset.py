from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn import svm, datasets, metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

digits = datasets.load_digits()

img_and_lbl = list(zip(digits.images, digits.target))

n_samples = len(digits.images)
data = digits.images.reshape((n_samples,-1))

classifier = svm.SVC(gamma=0.001)

trainTestSplit = int(n_samples*0.75)
classifier.fit(data[:trainTestSplit],digits.target[:trainTestSplit])

expected = digits.target[trainTestSplit:]
predicted = classifier.predict(data[trainTestSplit:])

print("\nConfusion Matrix: \n%s" % metrics.confusion_matrix(expected,predicted))
print("\nAccuracy: ",accuracy_score(expected,predicted))

plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation='nearest')
print("\nPrediction for the test image: ", classifier.predict(data[-2].reshape(1,-1)))

plt.show()
