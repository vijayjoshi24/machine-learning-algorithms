# -*- coding: utf-8 -*-
"""

@author: Vijay Joshi

"""

import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plotGraph

learningRates = [0.00001, 0.001, 0.1]

inputDataPerPeceptron = 785
totalOutputSize = 10
totalNoTrainingData = 60000
totalNoValidationData = 10000
totalNoOfEpochs = 50


# This utility function takes reference of Training and Validation MNIST data sets
# It furthers uses np.loadTxt utility to convert input data into ndDataArray
# And splitting Data and Labels into respective arrays
# Preprocessing of dataset is done by dividing with 255
def uploadDataSets(csvFileReference):
    ndDataArray = np.loadtxt(csvFileReference, delimiter=',')
    inputDataSet = np.insert(ndDataArray[:, np.arange(1, inputDataPerPeceptron)] / 255, 0, 1, axis=1)
    inputDataLabels = ndDataArray[:, 0]
    return inputDataSet, inputDataLabels


# Perceptron Training Function
# Calculation on weight adjustment against Training dataset
def perceptronTraining(weight):
    for i in range(0, totalNoTrainingData):
        opArray = np.dot(np.reshape(ipTrainingData[i, :], (1, inputDataPerPeceptron)), weight)
        predictedOP = np.insert(np.zeros((1, totalOutputSize - 1)), np.argmax(opArray), 1)
        targetOP = np.insert(np.zeros((1, totalOutputSize - 1)), int(ipTrainingLabels[i]), 1)
        error = targetOP - predictedOP
        # Calculating Adjusted delta of weight based on Error determined in above steps and input training datasets
        adjustedWeight = np.dot(np.reshape(ipTrainingData[i, :], (inputDataPerPeceptron, 1)),
                                np.reshape(error, (1, totalOutputSize)))
        weight += (currLearningRate * adjustedWeight)

    return weight


def perceptronValidating(inputDataset, inputLabels, totalSize):
    predictedOP = []
    for i in range(0, totalSize):
        opArray = np.dot(np.reshape(inputDataset[i, :], (1, inputDataPerPeceptron)), weight)
        predictedOP.append(np.argmax(opArray))

    return accuracy_score(inputLabels, predictedOP), predictedOP


# Loading Respective Training and Validation Data MNIST CSV
# Assigning respective Data Sets to Data Variables and Labels
print("Upload MNIST Training Data Set and assigning it to training Data and Target Label Set")
ipTrainingData, ipTrainingLabels = uploadDataSets('mnist_train.csv')
print("Upload MNIST Validation Data Set and assigning it to validation Data and Target Label Set")
ipValidationData, ipValidationLabels = uploadDataSets('mnist_validation.csv')

for currLearningRate in learningRates:

    currentEpochRun = 0
    epochArray = []
    validationAccuracyData = []
    trainingAccuracyData = []

    # Generating Random Weights
    weight = (np.random.rand(inputDataPerPeceptron, totalOutputSize) - 0.5) * (0.1)
    print("Randomized Weights: " + str(weight))

    while (1):
        # Validating Perceptron on Training Dataset and capturing its accuracy
        currentAccuracy, predictedOPList = perceptronValidating(ipTrainingData, ipTrainingLabels, totalNoTrainingData)
        print("Executing Epoch " + str(currentEpochRun))
        print("Accuracy around Training Datasets : " + str(currentAccuracy))
        if currentEpochRun == totalNoOfEpochs:
            break
            # Validating Perceptron on Validation Dataset and capturing its accuracy
        validationAccuracy, predictedOPList = perceptronValidating(ipValidationData, ipValidationLabels,
                                                                   totalNoValidationData)
        print("Accuracy around Validation Datasets: " + str(validationAccuracy))
        previousAccuracy = currentAccuracy
        currentEpochRun += 1
        # Invoking Perceptron Training function
        # With adjusted weights
        weight = perceptronTraining(weight)
        epochArray.append(currentEpochRun)
        trainingAccuracyData.append(currentAccuracy)
        validationAccuracyData.append(validationAccuracy)
    # Validation Perceptron post weight adjusted on training data execution
    validationAccuracy, predictedOPList = perceptronValidating(ipValidationData, ipValidationLabels,
                                                               totalNoValidationData)

    print("Accuracy around Training Datasets :" + str(validationAccuracy))
    print("Current Executing Learning Rate: " + str(currLearningRate))
    # Printing Confustion Matrix
    print(confusion_matrix(ipValidationLabels, predictedOPList))

    # Plotting Learning Rate graphs against Accuracy for Traning and Validation Datasets
    plotGraph.title("Learning Rate " + str(currLearningRate))
    plotGraph.plot(trainingAccuracyData)
    plotGraph.plot(validationAccuracyData)
    plotGraph.xlabel("Epochs")
    plotGraph.ylabel("Dataset Accuracy")

    plotGraph.show()


