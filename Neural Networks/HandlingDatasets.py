from pybrain.datasets import SupervisedDataSet

data = SupervisedDataSet(2,1)

data.addSample((0,0),(0))
data.addSample((1,0),(1))
data.addSample((0,1),(1))
data.addSample((1,1),(0))

for inp, tar in data:
    print("Input: ",inp, " and output: ",tar)

print(data['input'])
print(data['target'])
