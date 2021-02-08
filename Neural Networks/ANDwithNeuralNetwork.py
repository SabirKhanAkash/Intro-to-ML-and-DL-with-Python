from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

neuralNetwork = buildNetwork(2,3,1, bias=True)

dataset = SupervisedDataSet(2,1)

dataset.addSample((0,0),(0))
dataset.addSample((1,0),(0))
dataset.addSample((0,1),(0))
dataset.addSample((1,1),(1))

trainer = BackpropTrainer(neuralNetwork, dataset=dataset, learningRate=0.01, momentum=0.06)

for i in range(1,10000):
    error = trainer.train()

    if i%10000==0:
        print("Error in iteration ",i," is: ",error)
        print(neuralNetwork.activate([0,0]))
        print(neuralNetwork.activate([1,0]))
        print(neuralNetwork.activate([0,1]))
        print(neuralNetwork.activate([1,1]))

print("\n\nFinal result of XOR:\n")
print(neuralNetwork.activate([0,0]))
print(neuralNetwork.activate([1,0]))
print(neuralNetwork.activate([0,1]))
print(neuralNetwork.activate([1,1]))
