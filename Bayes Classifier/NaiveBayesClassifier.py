from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pd

data = pd.read_csv(
    "D:\\Study Materials\\Study\\My Python Workspace\\Intro to ML DL with Python\\Bayes Classifier\\credit_data.csv")

data.features = data[["income", "age", "loan"]]
data.target = data.default

feature_train, feature_test, target_train, target_test = train_test_split(data.features, data.target,
                                                                          test_size=0.3)

model = GaussianNB()
fittedmodel = model.fit(feature_train, target_train)
predictions = fittedmodel.predict(feature_test)

print("Confusion Matrix:\n", confusion_matrix(target_test, predictions))
print("\nAccuracy: ", accuracy_score(target_test, predictions))
