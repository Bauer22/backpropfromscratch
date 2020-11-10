'''
This is a from scratch, implementation of a backpropogation ANN, with momentum

It uses the MNIST data set, although it is coded to be very general and easy to change for other data

After 30 Epochs it reaches a 97% classification accuracy on the trg data
And 95% accuracy on the test data with the current parameters into the model
'''

#Imports
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
import itertools


# For this problem we have 784 input, 64 hidden and 10 output
#784 is a 28x28 pixel image flattened into an input vector
#10 output nodes for numbers 0-9
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

    # Generates random weights for the problem
    def genWeights(self, n, m):
        return np.random.randn(n, m)

    # Sigmoid Activation Function
    # Sigmoid is common, but there are lots of different functions you can use
    def sigmoid(self, a):
        a = 1 / (1 + np.exp(-a))
        return a

    # This FeedForward is for testing only
    # Feed forward simply does the inputs * weights
    # The feedforward of the back propogation is
    # hard coded into the train weights function
    def feedForward(self, x):
        # Turn 28x28 into 784x1
        sample = x.flatten()
        #Inputs x I->H weights
        #Then take the weighted sum and add the respective bias
        hidden = np.dot(sample, self.IHweights)   + self.Hbias
        # Find the output of the hidden nodes by using the activation function
        hidden = self.sigmoid(hidden)
        # Hidden Outputs x H->O Weights
        #Then Take the weighted sum and add the respective bias
        predicted = np.dot(hidden, self.HOweights)  + self.Obias
        # Find the output by using the sigmoid activation function
        predicted = self.sigmoid(predicted)
        #Return the one hot encoded vector
        return np.asarray(predicted)

    # Function that runs the testing
    # Includes the outputs for testing
    def test(self,test):
        predictions = []
        #If testing data, change outputs to reflect
        if test:
            x = self.testInput
            y = self.testReal
            z = "Confusion Matrix for Test Values"
        # If Training data, change outputs to reflect
        else:
            x = self.data
            y = self.realValue
            z = "Confusion Matrix for Training Values"

        #Feedforward the test values, create the array called predicted of all the values
        for i in range(len(x)):
            predictions.append(self.feedForward(x[i]))
        #Prepare the array to be put through the metrics functions
        predictions = np.asarray(predictions)
        predictions = predictions.reshape(len(predictions),10)

        #Classification report
        cm1 = classification_report(y.argmax(axis=1), predictions.argmax(axis=1))
        #Confusion matrix instantiation
        cm = confusion_matrix(y.argmax(axis=1), predictions.argmax(axis=1))
        #Overall Accuracy Reporting
        acc = accuracy_score(y.argmax(axis=1), predictions.argmax(axis=1, ),normalize=False)
        print("The accuracy is: ",acc,"/",len(y), " or: ", acc/len(y))
        print("Classification Report")
        print(cm1)
        plt.figure()
        #Format the confusion matrix
        plot_confusion_matrix(cm, classes=['0', '1', '2','3','4','5','6','7','8','9'],title=z)
        plt.show()

    # Function to train the weights of the ANN
    # This includes backpropogation
    def trainWeights(self):
        # This is for momentum
        previousDeltaWeightHO = []
        prevDbiasHO = []
        previousDeltaWeightIH = []
        prevDbiasIH = []

        for epoch in range(100):
            numCorrect = 0
            # For a sample
            for (x, y) in zip(self.data, self.realValue):
                # Turn 28x28 into 784x1
                sample = x.flatten()
                # one sample
                # Inputs x I->H weights
                # Then take the weighted sum and add the respective bias
                hidden = np.dot(sample, self.IHweights) + self.Hbias
                # Find the output of the hidden nodes by using the activation function
                hidden = self.sigmoid(hidden)
                # Hidden Outputs x H->O Weights
                # Take the weighted sum and add the respective bias
                predicted = np.dot(hidden, self.HOweights) + self.Obias
                # Find the output by using the sigmoid activation function
                predicted = self.sigmoid(predicted)
                real = y
                # Calculate the error
                error = real - predicted
                error = error.transpose()
                # Take the output of the ANN and find its specific prediction
                if np.argmax(predicted) == np.argmax(real):
                    numCorrect += 1

                # ------------- Back Prop Begins--------

                # Hidden to Output Layer

                # Derivative of the sigmoid. (y*(y-1)
                # So in this case take the output of the output nodes
                # Since they just came out of the sigmoid function
                # Then multiply the gradient by the calculared error
                gradient = error.T * predicted * (1 - predicted)
                #Multiply in the learning rate
                gradient = gradient * self.learningRate
                #Change in H -> O weights is the hidden output * the gradient
                deltaWeightHO = np.dot(hidden.transpose(), gradient)
                # Update the bias
                deltaOBias = self.Obias * gradient
                # Implementation of momentum & updating weights
                # alpha is 0.01
                # 0 is for first run
                if len(previousDeltaWeightHO) != 0:
                    self.HOweights = self.HOweights + deltaWeightHO + (0.01*previousDeltaWeightHO)
                    self.Obias = self.Obias + deltaOBias + (0.01*prevDbiasHO)
                else:
                    self.HOweights = self.HOweights + deltaWeightHO
                    self.Obias = self.Obias + deltaOBias

                # Input To Hidden

                #Calculate hidden errors by output errors and H -> O weights
                hiddenErrors = np.dot(self.HOweights, error)
                #Derivative of sigmoid using hidden output
                # Multiply in the errors to the gradient
                hiddenGradient = hiddenErrors.T * hidden * (1 - hidden)
                #Multiply in the learning rate
                hiddenGradient = hiddenGradient * self.learningRate
                sample = sample.reshape(784, 1)
                #Calculate the change in weights by multiplying the inputs by the gradient
                deltaWeightIH = np.dot(sample, hiddenGradient)
                # Update bias
                deltaHBias = self.Hbias * hiddenGradient

                # Implementation of momentum & updating weights
                # alpha is 0.01
                # None is for first run
                if len(previousDeltaWeightIH) != 0:
                    self.IHweights = self.IHweights + deltaWeightIH + (0.1 * previousDeltaWeightIH)
                    self.Hbias = self.Hbias + deltaHBias + (0.1 * prevDbiasIH)
                else:
                    self.IHweights = self.IHweights + deltaWeightIH
                    self.Hbias = self.Hbias + deltaHBias
            print("Epoch number: ", epoch, "Correct number: ", numCorrect / (len(self.data)))
            #If we end up with a 100% success before the number of epochs is over
            if ((numCorrect / (len(self.data)) >0.99 )):
                break

#I TOOK THIS CODE FROM https://datascience.stackexchange.com/questions/40067/confusion-matrix-three-classes-python
#This code simply takes a confusion matrix and format/colourizes it
def plot_confusion_matrix(cm, classes, title,normalize=True, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def main():
    # Preprocessing
    (x_train, y_train_raw), (x_test, y_test_raw) = mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    y_train = np.zeros((y_train_raw.shape[0], 10))

    for i in range(y_train.shape[0]):
        y_train[i][y_train_raw[i]] = 1
    y_test = np.zeros((y_test_raw.shape[0], 10))
    for i in range(y_test.shape[0]):
        y_test[i][y_test_raw[i]] = 1
    print("done preprocess")

    #Create the ANN
    test = ANN(x_train, y_train, x_test, y_test, 0.01, 784, 64, 10)
    test.trainWeights()
    #Do Training Outputs
    test.test(False)
    #Do Test Outputs
    test.test(True)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
