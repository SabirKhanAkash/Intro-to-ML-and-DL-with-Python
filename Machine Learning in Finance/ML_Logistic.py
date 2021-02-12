import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import sklearn
import pandas_datareader.data as web
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

def create_dataset(stock_symbol, startDate, endDate, lags=5):
    df = web.DataReader(stock_symbol, "yahoo", startDate, endDate)
    tslag = pd.DataFrame(index=df.index)
    tslag["Today"]= df["Adj Close"]

    for i in range(0,lags):
        tslag["lag%s" % str(i+1)] = df["Adj Close"].shift(i+1)

    dfret = pd.DataFrame(index=tslag.index)
    dfret["Today"] = tslag["Today"].pct_change()*100.0

    for i in range(0,lags):
        dfret["Lag%s" % str(i+1)] = tslag["lag%s" % str(i+1)].pct_change()*100.0

    dfret["Direction"] = np.sign(dfret["Today"])
    dfret.drop(dfret.index[:5], inplace=True)

    return dfret

if __name__ == "__main__":
    data = create_dataset("AAPL", datetime(2012,1,1), datetime(2017,5,31), lags=5)
    x = data[["Lag1","Lag2","Lag3","Lag4"]]
    y = data["Direction"]

    startTest = datetime(2017,1,1)

    xTrain = x[x.index < startTest]
    xTest = x[x.index >= startTest]
    yTrain = y[y.index < startTest]
    yTest = y[y.index >= startTest]

    model = LogisticRegression()

    model.fit(xTrain,yTrain)

    pred = model.predict(xTest)

    data.hist(bins=100, color="Green")
    plt.show()

    print("\nAccuracy of logistic regression model:\n ", model.score(xTest, yTest))
    print("\nConfusion Matrix: \n", confusion_matrix(pred,yTest))
