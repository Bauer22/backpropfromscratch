'''
This an implementation of a backpropogation ANNs using predefined libraries

It uses the MNIST data set.


'''


from sklearn.neural_network import MLPClassifier
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import itertools

#Load the data set and preprocess the images.
def load_dataset():
    (X_train, y_train_raw), (X_test, y_test_raw) = mnist.load_data()
    # normalize x
    X_train = X_train / 255.0
    X_test = X_test/ 255.0

    # One hot encoding
    # I.e. take the y value of 1 and turn in into [0,1,0,0,0...]
    y_train=np.zeros((y_train_raw.shape[0],10))
    for i in range(y_train.shape[0]):
        y_train[i][y_train_raw[i]]=1
    y_test=np.zeros((y_test_raw.shape[0],10))
    for i in range(y_test.shape[0]):
        y_test[i][y_test_raw[i]]=1
    return X_train, y_train, X_test, y_test

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

#Runs the model
#Improve changes the optimizer from SGD to Adam
def run():
        train_x, y_train, test_x, y_test = load_dataset()

        train_x = train_x.reshape(60000, 784)
        test_x = test_x.reshape(10000, 784)


        #This is creating the MLP. Sort of like the constructors from the fromScratch file
        model_sklearn = MLPClassifier(max_iter=30, hidden_layer_sizes=64, activation='logistic', learning_rate_init=0.01, )



        #This is the same as the trainWeights function
        model_sklearn.fit(train_x, y_train)
        print("Done Training")

        #This acts like the feedforeward function
        #Feeds foreward the training data one last time
        pred_y_train_sklearn = model_sklearn.predict(train_x)
        #Feeds foreward the test data one last time
        pred_y_test_sklearn = model_sklearn.predict(test_x)

        # -------- Training Data-------------
        #Classification Report
        trainingCR = classification_report(y_train.argmax(axis=1), pred_y_train_sklearn.argmax(axis=1))
        #Confusion Matrix
        trainingCM = confusion_matrix(y_train.argmax(axis=1), pred_y_train_sklearn.argmax(axis=1))
        #Accuracy Reporting
        acc = accuracy_score(y_train.argmax(axis=1), pred_y_train_sklearn.argmax(axis=1), normalize=False)
        title = "Training Data"
        title2 = "Test Data"


        #FORMAT OUTPUTS
        print("-------Training Data----------")
        print("The accuracy is: ", acc, "/", len(y_train), " or: ", acc / len(y_train))
        print("Classification Report")
        print(trainingCR)
        plt.figure()
        plot_confusion_matrix(trainingCM, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], title=title)
        plt.show()

        # -------- Testing Data-------------


        # Classification Report
        cr = classification_report(y_test.argmax(axis=1), pred_y_test_sklearn.argmax(axis=1))
        #Confusion Matrix
        cm2 = confusion_matrix(y_test.argmax(axis=1), pred_y_test_sklearn.argmax(axis=1))
        #Accuracy Reporting
        acc2 = accuracy_score(y_test.argmax(axis=1), pred_y_test_sklearn.argmax(axis=1), normalize=False)
        print("-------Testing Data----------")
        print("The accuracy is: ", acc2, "/", len(y_test), " or: ", acc2 / len(y_test))
        print("Classification Report")
        print(cr)
        plt.figure()
        plot_confusion_matrix(cm2, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], title=title2)
        plt.show()


def main():

    run()


if __name__ == '__main__':
    main()
