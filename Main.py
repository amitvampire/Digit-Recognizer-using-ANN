'''''''''
Project: Digit recognition using sigmoid neural network using MNIST data from SCRATCH!
AMIT KUMAR
IIT(ISM) Dhanbad
CSE(B.tech)
'''''''''

import NeuralNetwork
import LoadData

training_data, validation_data, test_data = LoadData.ProcessData()
neuron = NeuralNetwork.SigmoidNeuralNetwork([784, 30, 10])
neuron.MBGD(training_data, 25, 10, 3.0, test_data = test_data)
#mini-batch gradient descent
