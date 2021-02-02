import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

dataset = pd.read_csv("D:\\Study Materials\\Study\\My Python Workspace\\Intro to ML DL with Python Udemy Course Codes\\Linear Regression\\house_prices.csv")
size = dataset['sqft_living']
price = dataset['price']

x = np.array(size).reshape(-1,1)
y = np.array(price).reshape(-1,1)

model = LinearRegression()
model.fit(x,y)

regression_model_mse = mean_squared_error(x,y)
print("MSE = ",math.sqrt(regression_model_mse))
print("R squared value: ",model.score(x,y))

print(model.coef_[0])

print(model.intercept_[0])

plt.scatter(x, y, color='green')
plt.plot(x, model.predict(x), color='black')
plt.title("Linear Regression")
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()

print("Prediction by the model: ",model.predict([[1600]]))

