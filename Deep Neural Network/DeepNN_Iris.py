from keras.models import Sequential
from keras.layers import Dense
from pip._vendor.toml import encoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

dataset = load_iris()

features = dataset.data
targets = encoder.fit_transform(y)

trainFeatures, testFeatures, trainTargets, testTargets = train_test_split(features,targets,test_size=0.2)

model = Sequential()

model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10,input_dim=10, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(3,activation='softmax'))

optimizer = Adam(lr=0.005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(trainFeatures, trainTargets, epochs=1000, batch_size=20, verbose=2)

results = model.evaluate(testFeatures,testTargets)

print("Accuracy on the test dataset: %.2f" % results[1])
