import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  accuracy_score

data = pd.read_csv("D:\Study Materials\Study\My Python Workspace\Intro to ML DL with Python\Decision Trees\iris_data.csv")

data.features = data[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
data.targets = data.Class

feature_train, feature_test, target_train, target_test = train_test_split(data.features,data.targets, test_size=0.3)

model = DecisionTreeClassifier(criterion='entropy')
model.fitted = model.fit(feature_train, target_train)
model.predictions = model.fitted.predict(feature_test)

print("\nConfusion Matrix:\n",confusion_matrix(target_test, model.predictions))
print("\nAccuracy: ",accuracy_score(target_test,model.predictions))
