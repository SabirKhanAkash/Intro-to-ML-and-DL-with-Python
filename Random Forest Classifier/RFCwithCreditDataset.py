import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  accuracy_score

credit_data = pd.read_csv("D:\Study Materials\Study\My Python Workspace\Intro to ML DL with Python\Random Forest Classifier\credit_data.csv")

features = credit_data[["income", "age","loan"]]
targets = credit_data.default

feature_train, feature_test, target_train, target_test = train_test_split(features,targets,test_size=0.3)

model = RandomForestClassifier(n_estimators=1000, max_features='sqrt')
fitted_model = model.fit(feature_train, target_train)
predictions = fitted_model.predict(feature_test)

print("\nThe confusion Matrix:\n",confusion_matrix(target_test,predictions))
print("\nAccuracy: ",accuracy_score(target_test,predictions))
