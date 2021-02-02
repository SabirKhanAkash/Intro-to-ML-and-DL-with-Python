import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

CreditData = pd.read_csv("D:\Study Materials\Study\My Python Workspace\Intro to ML DL with Python Udemy Course Codes\Logistic Regression\credit_data.csv")

print(CreditData.head(),"\n-------------------------------------------------------------------------\n")
print(CreditData.describe(),"\n------------------------------------------------------------------------------\n")
print(CreditData.corr(),"\n-------------------------------------------------------------------------------\n")

features = CreditData[["income", "age", "loan"]]
target = CreditData.default

feature_train, feature_test, target_train, target_test = train_test_split(features,target,test_size=0.3)

model = LogisticRegression()
model.fit = model.fit(feature_train, target_train)
predictions = model.fit.predict(feature_test)

print("Confusion Matrix: \n",confusion_matrix(target_test, predictions))
print("\nAccuracy level of the model = ",accuracy_score(target_test, predictions))
