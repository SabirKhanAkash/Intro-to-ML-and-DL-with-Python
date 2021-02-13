import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

trainingData = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

targetData = np.array([[0],[1],[1],[0]], "float32")

model = Sequential()

model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

model.fit(trainingData, targetData, epochs=1000, verbose=2)

print("The predictions of the model: ",model.predict(trainingData).round())
