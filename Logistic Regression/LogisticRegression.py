from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

x1 = np.array([0,0.6,1.1,1.5,1.8,2.5,3,3.1,3.9,4,4.9,5,5.1])
y1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])

x2 = np.array([3,3.8,4.4,5.2,5.5,6.5,6,6.1,6.9,7,7.9,8,8.1])
y2 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])

x = np.array([[0],[0.6],[1.1],[1.5],[1.8],[2.5],[3],[3.1],[3.9],[4],[4.9],[5],[5.1],[3],[3.8],[4.4],[5.2],[5.5],[6.5],[6],[6.1],[6.9],[7],[7.9],[8],[8.1]])
y = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1])

plt.plot(x1, y1, 'ro', color='red')
plt.plot(x2, y2, 'ro', color='green')

model = LogisticRegression()
model.fit(x,y)

print("b0 = ", model.intercept_)
print("b1 = ", model.coef_)

def logistic(classifier, x):
    return 1/(1+np.exp(-(model.intercept_ + model.coef_ * x)))

for i in range(1,120):
    plt.plot(i/10.0-2, logistic(model,i/10.0),'ro', color = 'grey')

plt.axis([-2,10,-0.5,2])

plt.show()

pred0 = model.predict([[2]])
print("Prediction by the model for 2K BDT balance: ",pred0," -> means the person is not eligible for loan")

pred1 = model.predict([[6]])
print("Prediction by the model for 6K BDT balance: ",pred1," -> means the person is eligible for loan")
