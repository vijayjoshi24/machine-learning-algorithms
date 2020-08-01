# -*- coding: utf-8 -*-
"""

@author: Vijay Joshi

"""

import numpy as np
import math

class naiveBayes:

    def __init__(self, ipTrainingFile, ipTestFile):
        self.ipTrainingFile = ipTrainingFile
        self.ipTestFile = ipTestFile
        self.ipTrainingData = []
        self.ipTestData = []
        self.classLabels = []
        self.classes = []
        self.dimensions = 0

    def uploadData(self):

        # Reading Training and Test data file
        ipTrainingFile = open(self.ipTrainingFile, "r")
        ipTrainingFileLines = ipTrainingFile.readlines()

        ipTestFile = open(self.ipTestFile, "r")
        ipTestFileLines = ipTestFile.readlines()

        #Finding Dimensions based on first line
        #Later it will be subtracted with 1 as one of the dimension is for class label
        self.dimensions = len(list(filter(None, ipTrainingFileLines[1].split(" "))))
        print("Dimensions>>>>>"+ str(self.dimensions))

        # Arrays for Training, Test and Class Label datasets
        self.ipTrainingData = np.zeros((1, self.dimensions - 1))
        self.ipTestData = np.zeros((1, self.dimensions))
        self.classLabels = np.zeros((1, 1))

        for trainData in ipTrainingFileLines:
            trainDataRow = trainData.split(" ")
            trainDataRow = list(filter(None, trainDataRow))
            trainDataRow = list(map(float, trainDataRow))
            # Stacking Training Data and class Labels in arrays created
            self.ipTrainingData = np.vstack((self.ipTrainingData, trainDataRow[:-1]))
            self.classLabels = np.vstack((self.classLabels, trainDataRow[-1]))

        for testData in ipTestFileLines:
            testDataRow = testData.split(" ")
            testDataRow = list(filter(None, testDataRow))
            testDataRow = list(map(float, testDataRow))
            self.ipTestData = np.vstack((self.ipTestData, testDataRow))

        self.ipTrainingData = self.ipTrainingData[1:]
        self.ipTestData = self.ipTestData[1:]
        self.classLabels = self.classLabels[1:]
        self.classes = np.unique(self.classLabels)

    def classificationAccuracy(labels, groundTruth, probability):
        maximumProbability = np.max(probability)
        maxProbIndex = np.nonzero(probability == maximumProbability)[0]
        labelPredicted = 0
        for i in maxProbIndex:
            i = int(i)
            if labels[i] == groundTruth:
                classificationAccuracy = 1 / maxProbIndex.shape[0]
                labelPredicted = groundTruth
            else:
                classificationAccuracy = 0.0
        labelPredicted = int(labels[i])
        return labelPredicted, maximumProbability, classificationAccuracy

    def displayResults(result):
        counter = 0

        for i in range(result.shape[0]):
            print("ID={:5d}, predicted={:3d}, probability= {:.4f}, true={:3d}, accuracy= {:4.2f}".format(
                int(result[i, 0]),
                int(result[i, 1]),
                result[i, 2],
                int(result[i, 3]),
                result[i, 4]))
            if (result[i, 1] == result[i, 3]):
                counter += 1
        print("classification accuracy= {:6.4f}".format(counter / result.shape[0]))


class gaussianClassifier(naiveBayes):
    def __init__(self, ipTrainingFile, ipTestFile):
        naiveBayes.__init__(self, ipTrainingFile, ipTestFile)

    def normalProbabilityDensity(self, x, mu, sigma):
        u = (x - mu) / abs(sigma)
        y = (1 / (np.sqrt(2 * math.pi) * abs(sigma))) * math.exp(-u * u / 2)
        return y

    def gaussianTraining(self):
        print("Starting : Gaussian Training >>>>>>>>>>>>>>>>>")
        totalClasses = len(self.classes)
        print("Total classes>>>>>>" + str(totalClasses))
        dimensions = self.dimensions - 1
        self.gaussianDistribution = np.empty((totalClasses, dimensions, 2))


        for classLabel in self.classes:
            classLabel = int(classLabel)
            classLabelIndex = np.nonzero(self.classLabels == classLabel)[0]
            for dimension in range(dimensions):
                mean = np.mean(self.ipTrainingData[classLabelIndex, dimension])
                stdev = np.std(self.ipTrainingData[classLabelIndex, dimension])
                if stdev < 0.01:
                    stdev = 0.01
                print("Class {:d}, attribute {:d}, mean = {:.2f}, std = {:.2f}".format(classLabel, dimension, mean, stdev))
                print("Class label {:d}, Current Dimension {:d} ".format((classLabel - 1), dimension))
                self.gaussianDistribution[classLabel - 1, dimension, 0] = mean
                self.gaussianDistribution[classLabel - 1, dimension, 1] = stdev
                print("Classification >>>>>>>>>>>>>>>>>")

    def gaussianTesting(self):
        print("Gaussian Testing >>>>>>>>>>>>>>>>>")
        testDataSize = self.ipTestData.shape[0]
        trainDataSize = self.ipTrainingData.shape[0]
        totalClasses = len(self.classes)
        dimensions = self.dimensions - 1
        testDataProbability = np.zeros((testDataSize, totalClasses))
        probabilityOfClass = np.divide(np.histogram(self.classLabels, range=(1, 10))[0], trainDataSize)
        result = np.empty((testDataSize, 5))

        for i in range(testDataSize):
            for j in range(totalClasses):
                probability = 1
                for dimension in range(dimensions):
                    probability = probability * self.normalProbabilityDensity(self.ipTestData[i, dimension], self.gaussianDistribution[j, dimension, 0],
                                                             self.gaussianDistribution[j, dimension, 1])
                testDataProbability[i, j] = probability * probabilityOfClass[j]
            testDataProbabilitySum = np.sum(testDataProbability[i, :])
            for j in range(totalClasses):
                if testDataProbabilitySum == 0:
                    testDataProbability[i, j] = 1 / totalClasses
                else:
                    testDataProbability[i, j] = np.divide(testDataProbability[i, j], testDataProbabilitySum)

            result[i, 0] = i + 1
            result[i, 3] = self.ipTestData[i, -1]
            result[i, 1], result[i, 2], result[i, 4] = naiveBayes.classificationAccuracy(labels=self.classes,
                                                                                 groundTruth=self.ipTestData[i, -1],
                                                                                 probability=testDataProbability[i, :])

        naiveBayes.displayResults(result)

def main():

    print("Provide the training & test data's file path in following format: ")
    print("<absolute_training_file_path_location> <absolute_test_file_path_location>")
    inputFileLocation = input()
    fileLocationPath = inputFileLocation.split()
    if (len(fileLocationPath) > 1):
        gaussian = gaussianClassifier(ipTrainingFile=fileLocationPath[0], ipTestFile=fileLocationPath[1])
        gaussian.uploadData()
        gaussian.gaussianTraining()
        gaussian.gaussianTesting()
    else:
        print("File location doesn't contain both file location")


main()