# Alex Zhu
# This was written for Python 3.4.2
from __future__ import division
import numpy as np
import time

class neuralnet:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, momentum):
        # init input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # init weights from -0.05 to 0.05
        self.wih = np.random.normal(0.0, 0.05, (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, 0.05, (self.onodes, self.hnodes))

        # init learning rate and momentum const
        self.lr = learningrate
        self.mc = momentum

        # init sigmoid function for the activation
        self.sigmoid_function = lambda x: 1 / (1 + np.exp(-x))


    # train function
    # inputs are inputs, targets, previous change in input to hidden and change to hidden to out
    def train(self, inputs_arr, targets_list, prevdxih, prevdxho):

        inputs = np.array(inputs_arr, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # inputs to hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate output from hidden layer w/ sigmoid
        hidden_outputs = self.sigmoid_function(hidden_inputs)

        # hidden layer output to final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate output from final output layer w/ sigmoid
        final_outputs = self.sigmoid_function(final_inputs)

        # output layer error: difference between target and final
        output_errors = targets - final_outputs
        # hidden layer error: output_errors, split by weights, dot with hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # These are the change in weights as recorded for next train; are output of train
        olddxih = self.lr * np.dot((hidden_errors * (hidden_outputs - hidden_outputs**2)), np.transpose(inputs)) + self.mc*prevdxih

        olddxho = self.lr * np.dot((output_errors * (final_outputs - final_outputs**2)), np.transpose(hidden_outputs)) + self.mc*prevdxho


        # update weights between input and hidden layers w/ momentum
        self.wih += self.lr * np.dot((hidden_errors * (hidden_outputs - hidden_outputs**2)), np.transpose(inputs)) + self.mc*prevdxih
        # update weights between hidden and output layers w/ momentum
        self.who += self.lr * np.dot((output_errors * (final_outputs - final_outputs**2)), np.transpose(hidden_outputs)) + self.mc*prevdxho


        return olddxih, olddxho


    def compute(self, inputs_arr):
        # Identical to how train works only w/o weight updating
        inputs = np.array(inputs_arr, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.sigmoid_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.sigmoid_function(final_inputs)

        return final_outputs

if __name__ == '__main__':
    input_nodes = 785
    hidden_nodes = 100
    output_nodes = 10
    record_limit = 60000
    epochs = 50
    learning_rate = 0.1
    momentum = 0.25
    start = time.time()
    nn = neuralnet(input_nodes, hidden_nodes, output_nodes, learning_rate, momentum)

    # load mnist training data
    training_data_file = open("mnist_train.csv", 'r')

    training_data_list = training_data_file.readlines()
    training_data_file.close()



    test_data_file = open("mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    print("MNIST training and testing data loaded")

    for e in range(epochs):

        record_index = 0
        # previous change in w
        pdxih = 0
        pdxho = 0
        # variable input of prev change in w
        pdxihin = 0
        pdxhoin = 0
        correctTrainingRecords = 0
        numTrainingRecords = record_limit
        correctTestRecords = 0
        numTestRecords = 0
        for record in training_data_list:
            # format remove commas

            records = record.split(',')
            # append a bias input
            # num = 255 b/c I'm lazy and wanted to do the scale to the whole record
            records.append(255.0)

            inputs = (np.asfarray(records[1:]) / 255.0)

            targets = np.zeros(output_nodes) + 0.1

            # records[0] = label of the image, where value target is 0.9
            targets[int(records[0])] = 0.9

            # Making sure train runs
            # if record_index % 5000 == 0:
            #     print("training on row", record_index)
            if record_index == record_limit:  # testing code w/o going through all records, also for experiments
                break
            record_index += 1
            pdxih, pdxho = nn.train(inputs, targets, pdxihin, pdxhoin)
            # Calculate accuracy on training set
            correct_training_label = int(records[0])
            training_outputs = nn.compute(inputs)
            training_label = np.argmax(training_outputs)
            if training_label == correct_training_label:
                # Num correct training set guesses
                correctTrainingRecords += 1

            pdxihin = pdxih
            pdxhoin = pdxho
        # calculate test set accuracy in epoch n
        for record in test_data_list:
            # then split the record by the ',' commas
            records = record.split(',')
            # append a bias input
            records.append(255.0)
            # do note correct answer is first value
            correct_label = int(records[0])

            inputs = (np.asfarray(records[1:]) / 255.0)

            # run nn on data
            outputs = nn.compute(inputs)

            # max of outputs is the discovered label
            label = np.argmax(outputs)

            # now, append correct or incorrect to accuracy list
            if (label == correct_label):
                # correct answer so add 1 to correctRecords
                correctTestRecords += 1
            numTestRecords += 1

        # Accuracy as floats
        train_accuracy = correctTrainingRecords / float(numTrainingRecords)
        test_accuracy = correctTestRecords / float(numTestRecords)
        print("Epoch: ", e)
        print("Train Accuracy: ", train_accuracy)
        print("Test Accuracy: ", test_accuracy)

    test_data_file = open("mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # matrices and variables to record data.

    correctRecords = 0
    numRecords = 0

    # Matrix size is equal to output nodes size
    confusion_matrix = np.zeros((output_nodes, output_nodes), dtype=int)

    # Run on entire test data
    for record in test_data_list:

        # then split the record by the ',' commas
        records = record.split(',')
        # append a bias input
        records.append(255.0)
        # do note correct answer is first value
        correct_label = int(records[0])

        inputs = (np.asfarray(records[1:]) / 255.0)

        # run nn on data
        outputs = nn.compute(inputs)

        # max of outputs is the discovered label
        label = np.argmax(outputs)

        # now, append correct or incorrect to accuracy list
        if (label == correct_label):

            # correct answer so add 1 to correctRecords
            correctRecords += 1
            # append value to confusion matrix indice [correct_label][label]
            confusion_matrix[correct_label][label] += 1

        else:

            # incorrect answer so don't add to correct Records
            # append value to confusion matrix indice [correct_label][label]

            confusion_matrix[correct_label][label] += 1

        numRecords += 1

    # Calculate accuracy, precision, recall
    accuracy = correctRecords / float(numRecords)
    # These are arrays with indexes equal to precision of number "n" from 0-9
    precision = confusion_matrix.sum(axis=0)
    recall = confusion_matrix.sum(axis = 1)

    # for i in range(output_nodes):
    #     precision[i] = confusion_matrix[i][i] / float(precision[i])
    #     recall[i] = confusion_matrix[i][i] / float(recall[i])

    # accuracy = num correct guesses / num of records
    print("Accuracy = ", accuracy)
    print(np.matrix(confusion_matrix))
    print("Precision Sums")
    print(np.matrix(precision))
    print("Recall Sums")
    print(np.matrix(recall))
    end = time.time()
    print("Time for Execution: ", (end - start))