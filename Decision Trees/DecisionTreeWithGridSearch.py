import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("D:\Study Materials\Study\My Python Workspace\Intro to ML DL with Python\Decision Trees\iris_data.csv")

data.features = data[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
data.targets = data.Class

paramGrid = {'max_depth': np.arange(1,10)}

tree = GridSearchCV(DecisionTreeClassifier(),paramGrid)

feature_train, feature_test, target_train, target_test = train_test_split(data.features,data.targets, test_size=0.2)

tree.fit(feature_train, target_train)
tree_predictions = tree.predict_proba(feature_test)[:,1]

print("\nbest parameter with Grid Search:\n",tree.best_params_)
