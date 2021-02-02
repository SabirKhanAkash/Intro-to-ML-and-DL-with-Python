import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

dataset = pd.read_csv("D:\Study Materials\Study\My Python Workspace\Intro to ML DL with Python Udemy Course Codes\K-Nearest Neighbour Classifier\credit_data.csv")

features = dataset[["income", "age", "loan"]]
target = dataset.default

features = preprocessing.MinMaxScaler().fit_transform(features)
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

model = KNeighborsClassifier(n_neighbors=20)
fittedModel = model.fit(feature_train, target_train)
predictions = fittedModel.predict(feature_test)

print("Confusion Matrix:\n", confusion_matrix(target_test, predictions))
print("\nAccuracy: ", accuracy_score(target_test,predictions))
