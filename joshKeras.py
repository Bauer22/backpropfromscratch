#Imports
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
import itertools

class ANN():
    def __init__(self, input, realValue, testInput, testReal, learningRate, inNum=784, hiddenNum=64, outputNum=10):
        self.IHweights = self.genWeights(inNum, hiddenNum)
        self.HOweights = self.genWeights(hiddenNum, outputNum)
        self.Hbias = self.genWeights(1, hiddenNum)
        self.Obias = self.genWeights(1, outputNum)
        self.outputNum = outputNum
        self.data = input
        self.realValue = realValue
        self.learningRate = learningRate
        self.testInput = testInput
        self.testReal = testReal


if __name__ == '__main__':
    main()
