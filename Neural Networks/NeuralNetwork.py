from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer, SoftmaxLayer

network = buildNetwork(2,3,1, bias=True)

print(network.activate([1,0]))
print(network['in'])
print(network['hiddem0'])
print(network['out'])
print(network['bias'])
