# -*- coding: utf-8 -*-
"""

@author: Vijay Joshi

"""

import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plotGraph

learningRate = 0.1
inputData = 785
totalOutputSize = 10
totalNoTrainingData = 60000
totalNoValidationData = 10000
totalNoOfEpochs = 50
hiddenUnits = [20, 50, 100]
momentumValue = [0.25, 0.5, 0.95]

# This utility function takes reference of Training and Validation MNIST data sets
# It furthers uses np.loadTxt utility to convert input data into ndDataArray
# And splitting Data and Labels into respective arrays
# Preprocessing of dataset is done by dividing with 255
def uploadDataSets(csvFileReference):
    ndDataArray = np.loadtxt(csvFileReference, delimiter=',')
    inputDataSet = np.insert(ndDataArray[:, np.arange(1, inputData)] / 255, 0, 1, axis=1)
    inputDataLabels = ndDataArray[:, 0]
    return inputDataSet, inputDataLabels

def sigmoidFunction(vals):
    return 1 / (1 + np.exp(-vals))

def derivativeSigmoid(l):
    return l * (1 - l)

def forwardPropagation(ip, weightInputHidden, weightHiddenOutput):
    inputArray = np.reshape(ip, (1, inputData))
    hiddenArray = sigmoidFunction(np.dot(inputArray, weightInputHidden))
    hiddenArray[0][0] = 1
    outputArray = sigmoidFunction(np.dot(hiddenArray, weightHiddenOutput))
    return inputArray, hiddenArray, outputArray

def backwardPropagation(error, inputArray, hiddenArray, outputArray, weightHiddenOutput, weightInputHidden, weightHiddenOutputPrev, weightInputHiddenPrev, mmtVal):
    adjustedOp = derivativeSigmoid(outputArray) * error
    adjustedOpHidden = derivativeSigmoid(hiddenArray) * np.dot(adjustedOp, np.transpose(weightHiddenOutput))
    weightHiddenOutputCurrent = (learningRate * np.dot(np.transpose(hiddenArray), adjustedOp)) + (mmtVal * weightHiddenOutputPrev)
    weightInputHiddenCurrent = (learningRate * np.dot(np.transpose(inputArray), adjustedOpHidden)) + (mmtVal * weightInputHiddenPrev)
    weightHiddenOutput = weightHiddenOutput + weightHiddenOutputCurrent
    weightInputHidden = weightInputHidden + weightInputHiddenCurrent
    return weightHiddenOutput, weightInputHidden, weightHiddenOutputCurrent, weightInputHiddenCurrent

def trainingNeuralNetwork(weightHiddenOutput, weightInputHidden, weightHiddenOutputPrev, weightInputHiddenPrev, mmtVal):
    for i in range(0, totalNoTrainingData):
        inputArray, hiddenArray, outputArray = forwardPropagation(ipTrainingData[i, :], weightInputHidden, weightHiddenOutput)
        kthTarget = np.insert((np.zeros((1, totalOutputSize - 1)) + 0.1), int(ipTrainingLabels[i]), 0.9)
        weightHiddenOutput, weightInputHidden, weightHiddenOutputPrev, weightInputHiddenPrev = backwardPropagation(kthTarget - outputArray, inputArray, hiddenArray, outputArray, weightHiddenOutput, weightInputHidden, weightHiddenOutputPrev, weightInputHiddenPrev, mmtVal)
    return weightInputHidden, weightHiddenOutput

def validatingNeuralNetwork(dataset, data_labels, set_size, weightInputHidden, weightHiddenOutput):
    predictedOPList = []
    for i in range(0, set_size):
        inputArray, hiddenArray, outputArray = forwardPropagation(dataset[i, :], weightInputHidden, weightHiddenOutput)
        predictedOPList.append(np.argmax(outputArray))
    return accuracy_score(data_labels, predictedOPList), predictedOPList

def neuralNetworkFunction(hiddenUnitSize, mmtVal):
    print("Starting : Neural Network Function")
    #Defining Randomized weights for Input to Hidden Layer
    weightInputHidden = (np.random.rand(inputData, hiddenUnitSize) - 0.5) * 0.1
    # Defining Randomized weights for Hidden to Output Layer
    weightHiddenOutput = (np.random.rand(hiddenUnitSize, totalOutputSize) - 0.5) * 0.1
    weightInputHiddenPrev = np.zeros(weightInputHidden.shape)
    weightHiddenOutputPrev = np.zeros(weightHiddenOutput.shape)
    validationAccuracyData = []
    trainingAccuracyData = []
    for epoch in range(0, 50):
        currentAccuracy, predictedOPList = validatingNeuralNetwork(ipTrainingData, ipTrainingLabels, totalNoTrainingData, weightInputHidden, weightHiddenOutput)
        validationAccuracy, predictedOPList = validatingNeuralNetwork(ipValidationData, ipValidationLabels, totalNoValidationData, weightInputHidden, weightHiddenOutput)
        print("Executing Current Epoch :" + str(epoch))
        print("Accuracy around Training Datasets : " + str(currentAccuracy))
        print("Accuracy around Validation Datasets : " + str(validationAccuracy))
        weightInputHidden, weightHiddenOutput = trainingNeuralNetwork(weightHiddenOutput, weightInputHidden, weightHiddenOutputPrev, weightInputHiddenPrev, mmtVal)
        trainingAccuracyData.append(currentAccuracy)
        validationAccuracyData.append(validationAccuracy)
    epoch = epoch + 1
    currentAccuracy, predictedOPList = validatingNeuralNetwork(ipTrainingData, ipTrainingLabels, totalNoTrainingData, weightInputHidden, weightHiddenOutput)
    validationAccuracy, predictedOPList = validatingNeuralNetwork(ipValidationData, ipValidationLabels, totalNoValidationData, weightInputHidden, weightHiddenOutput)
    print("Executing Current Epoch :" + str(epoch))
    print("Accuracy around Training Datasets : " + str(currentAccuracy))
    print("Accuracy around Validation Datasets : " + str(validationAccuracy))
    print("Input Hidden Layer Size : " + str(hiddenUnitSize))
    print("Input Momentum Value : " + str(mmtVal))
    print("Total number of input training datasets : " + str(totalNoTrainingData))
    print("Printing Confusion Matrix :")
    print(confusion_matrix(ipValidationLabels, predictedOPList))

    plotGraph.title("Learning Rate %r" % learningRate)
    plotGraph.plot(trainingAccuracyData)
    plotGraph.plot(validationAccuracyData)
    plotGraph.xlabel("Epochs")
    plotGraph.ylabel("Accuracy %")

    plotGraph.show()
    return

# Loading Respective Training and Validation Data MNIST CSV
# Assigning respective Data Sets to Data Variables and Labels
print("Upload MNIST Training Data Set and assigning it to training Data and Target Label Set")
ipTrainingData, ipTrainingLabels = uploadDataSets('mnist_train.csv')
print("Upload MNIST Validation Data Set and assigning it to validation Data and Target Label Set")
ipValidationData, ipValidationLabels = uploadDataSets('mnist_validation.csv')

# Experiment 1: Vary number of hidden units.
print("Execution of Experiment 1 : Vary number of hidden units")
for hiddenUnit in hiddenUnits:
    print("Execution with hidden unit: " +str(hiddenUnit))
    neuralNetworkFunction(hiddenUnit, 0.95)

# Experiment 2: Vary the number of training examples.
print("Execution of Experiment 2 : Vary the number of training examples")
for i in range(0, 2):
    ipTrainingData, X, ipTrainingLabels, Y = train_test_split(ipTrainingData, ipTrainingLabels, test_size=0.50)
    totalNoTrainingData = int(totalNoTrainingData / 2)
    neuralNetworkFunction(100, 0.95)

# Experiment 3: Vary the momentum value
print("Execution of Experiment 3 : Vary the momentum value")
for mmtVal in momentumValue:
    neuralNetworkFunction(100, mmtVal)
