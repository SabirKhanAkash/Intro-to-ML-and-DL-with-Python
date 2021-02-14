import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

pricesDatasetTrain = pd.read_csv('D:\Study Materials\Study\My Python Workspace\Intro-to-ML-and-DL-with-Python\Recurrent Neural Network\SP500_train.csv')
pricesDatasetTest = pd.read_csv('D:\Study Materials\Study\My Python Workspace\Intro-to-ML-and-DL-with-Python\Recurrent Neural Network\SP500_test.csv')

trainingset = pricesDatasetTrain.iloc[:,5:6].values
testset = pricesDatasetTest.iloc[:,5:6].values

min_max_scaler = MinMaxScaler(feature_range=(0,1))
scaled_trainingset = min_max_scaler.fit_transform(trainingset)

xTrain = []
yTrain = []

for i in range(40,1250):
    xTrain.append(scaled_trainingset[i-40:i,0])
    yTrain.append(scaled_trainingset[i,0])

xTrain = np.arrayx(xTrain)
yTrain = np.array(yTrain)

xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1],1))

model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(xTrain.shape[1],1)))
model.add(Dropout(0.5))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=50))
model.add(Dropout(0.3))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xTrain, yTrain, epochs=100, batch_size=32)

datasetTotal = pd.concat((pricesDatasetTrain['adj_close'], pricesDatasetTest['adj_close']), axis=0)
inputs = datasetTotal[len(datasetTotal)-len(pricesDatasetTest)-40:].values
inputs = inputs.reshape(-1,1)

inputs = min_max_scaler.transform(inputs)

xTest = []
for i in range(40,len(pricesDatasetTest)+40):
    xTest.append(inputs[i-40:i,0])

xTest = np.array(xTest)
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1],1))

predictions = model.predict(xTest)

predictions = min_max_scaler.inverse_transform(predictions)

plt.plot(testset, color='blue', label='Actual S&P500 Prices')
plt.plot(predictions, color='green', label='LSTM Predictions')
plt.title('S&P500 Predictions with Recurrent Neural Network')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

