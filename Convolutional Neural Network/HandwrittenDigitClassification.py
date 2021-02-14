import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
print("xTrain Original Shape: ",xTrain.shape)
print("yTrain Original Shape:",yTrain.shape)
print("xTest Original Shape:",xTest.shape)
print("yTest Original Shape:",yTest.shape)

plt.imshow(xTrain[0], cmap='gray')
plt.title('Class '+str(yTrain[0]))

featuresTrain = xTrain.reshape(xTrain.shape[0],28, 28, 1)
featuresTest = xTest.reshape(xTest.shape[0],28,28,1)

featuresTrain = featuresTrain.astype('float32')
featuresTest = featuresTest.astype('float32')

featuresTrain /= 255
featuresTest /= 255

targetTrain = np_utils.to_categorical(yTrain, 10)
targetTest = np_utils.to_categorical(yTest, 10)

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(28,28,1)))
model.add(Activation('relu'))

model.add(BatchNormalization())
model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add((MaxPooling2D(pool_size=(2,2))))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

model.compile(loss='cateorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(featuresTrain, targetTrain, batch_size=128, epochs=2, validation_data=(featuresTest,targetTest), verbose=1)

score = model.evaluate(featuresTest, targetTest)
print("Test accuracy: " % score[1])
