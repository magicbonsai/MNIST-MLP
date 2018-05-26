# MNIST-MLP
Multilayer Perceptron on the public MNIST database done for CS445 at Portland State University

# Overview
This is a simple implementation of a three layer multilayer Perceptron in Python.  The algorithm takes 768 inputs corresponding 
to the pixels in each image of a handwritten digit and runs them through a layer of 'n' hidden nodes, configurable by the program, 
and then to an output layer of 10 nodes, of which each node corresponds to a digit from 0 to 9.  The greatest output node value is 
considered in this implementation as the digit the classifier thinks the image is.

# Requirements
	- Python 3.4.2
		- External Libraries numpy, time
	- The MNIST training and testing datasets in '.csv' format.

This code can be run as long as the user possesses the 'mnist_test.csv', 'mnist_train.csv", and MNIST-MLP.py files and that they all exist 
within the same directory.