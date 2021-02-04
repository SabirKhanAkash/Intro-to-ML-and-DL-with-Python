from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import  accuracy_score
from sklearn import datasets
from sklearn.model_selection import GridSearchCV

dataset = datasets.load_digits()

image_features = dataset.images.reshape((len(dataset.images),-1))
image_targets = dataset.target

random_forest_model = RandomForestClassifier(n_jobs=-1, max_features='sqrt')

feature_train, feature_test, target_train, target_test = train_test_split(image_features,image_targets,test_size=0.3)

paramGrid = {
    "n_estimators" : [10,100,500,1000],
    "max_depth" : [1,5,10,15],
    "min_samples_leaf" : [1,2,3,4,5,10,15,20,30,40,50]
}

gridSearch = GridSearchCV(estimator=random_forest_model, param_grid=paramGrid, cv=10)
gridSearch.fit(feature_train, target_train)
print(gridSearch.best_params_)

optimalEstimators = gridSearch.best_params_.get("n_estimators")
optimalDepth = gridSearch.best_params_.get("max_depth")
optimalLeaf = gridSearch.best_params_.get("min_samples_leaf")

bestModel = RandomForestClassifier(n_estimators=optimalEstimators, max_depth=optimalDepth, max_features='sqrt', min_samples_leaf=optimalLeaf)
k_fold = model_selection.KFold(n_splits=10, random_state=123)

predictions = model_selection.cross_val_predict(bestModel, feature_test, target_test, cv=k_fold)
print("Accuracy of the tuned model: ", accuracy_score(target_test, predictions))
