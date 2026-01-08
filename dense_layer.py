"""
Objectives :

1. Create dense layer class
2.
3.
4.

"""

import numpy as np
from nnfs.datasets import spiral_data

class dense_layer():

    def __init__(self, inputs, neurons):
        self.weights = 0.01*np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

X, y = spiral_data(samples = 100, classes = 3)

dense = dense_layer(2, 3)

dense.forward(X)

print(dense.output[:5])

