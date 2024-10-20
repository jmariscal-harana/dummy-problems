# From https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
from random import random
from random import seed
import math


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = []
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for given input/weight pairs
def activate(weights, inputs):
    activation = weights[-1]  # Get bias first
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + math.exp(-activation))  # Sigmoid function


# Forward propagate input to a network output
def forward_propagate(network, inputs):
    for layer in network:
        outputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)  # wx + b
            neuron['output'] = transfer(activation)  # f(wx + b)
            outputs.append(neuron['output'])  
        inputs = outputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)  # Derivative of sigmoid function


# Propagate error to each layer and neuron
def backward_propagate_error(network, expected):
    network_len = len(network)
    for i in reversed(range(network_len)):
        layer = network[i]
        layer_len = len(layer)
        errors = []
        if i != network_len-1:
            for j in range(layer_len):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(layer_len):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(layer_len):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def main():
    # Initialise network
    seed(1)
    network = initialize_network(2, 3, 2)
    print("\nHIDDEN LAYER")
    [print("Neuron {0}:\n".format(i), neuron) for i, neuron in enumerate(network[0])]
    print("\nOUTPUT LAYER")
    [print("Neuron {0}:\n".format(i), neuron) for i, neuron in enumerate(network[1])]

    # Test forward propagation
    inputs = [0, 1]
    output = forward_propagate(network, inputs)
    print("\nOUTPUT (forward propagation):")
    print(output)

    # test backpropagation of error
    target = [0, 1]
    for _ in range(100):
        backward_propagate_error(network, target)
    for layer in network:
        print(layer)


if __name__ == '__main__':
    main()