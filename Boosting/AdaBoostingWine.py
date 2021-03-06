import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def isTasty(quality):
    if quality >= 7:
        return 1
    else:
        return 0;

data = pd.read_csv("D:\Study Materials\Study\My Python Workspace\Intro to ML DL with Python\Boosting\wine.csv", sep=";" )

features = data[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
data['tasty'] = data["quality"].apply(isTasty)
targets = data['tasty']

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.2)

paramDist = {
    'n_estimators': [50,100,200],
    'learning_rate': [0.01,0.05,0.1,0.3,1],
}

gridSearch = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=paramDist, cv=10)
gridSearch.fit(feature_train,target_train)

predictions = gridSearch.predict(feature_test)

print("\nConfusion Matrix: \n",confusion_matrix(target_test,predictions))
print("Accuracy: ",accuracy_score(target_test,predictions))
