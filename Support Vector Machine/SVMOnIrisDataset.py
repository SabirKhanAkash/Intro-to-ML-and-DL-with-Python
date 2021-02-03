import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets

dataset = datasets.load_iris()

features = dataset.data
targetVar = dataset.target

feature_train, feature_test, target_train, target_test = train_test_split(features, targetVar, test_size=0.3)

model = svm.SVC()
fittedModel = model.fit(feature_train,target_train)
predictions = fittedModel.predict(feature_test)

print("----------------------------------------\n\nWhen Gamma and C ar not defined")
print("\nConfusion Matrix:\n",confusion_matrix(target_test, predictions))
print("\nAccuracy: ", accuracy_score(target_test, predictions))

model = svm.SVC(gamma=0.001, C=100)
fittedModel = model.fit(feature_train,target_train)
predictions = fittedModel.predict(feature_test)

print("----------------------------------------\n\nWhen Gamma = 0.001 and C = 100")
print("\nConfusion Matrix:\n",confusion_matrix(target_test, predictions))
print("\nAccuracy: ", accuracy_score(target_test, predictions))
