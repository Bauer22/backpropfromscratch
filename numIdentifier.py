# THIS DEFINITELY DOES NOT WORK YET
# had lots of trouble with using the provided data - don't get how to input it
# tried to follow along with keras tutorial so everything is mostly from the cat dog example but
# once I get the data properly inputed I will probably have to change some stuff since instead of two
# options there are ten but I think the general idea is the same

from sklearn.neural_network import MLPClassifier
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import itertools


#  print("hi")
#Patrick's code for loading data and one hot encoding
# def load_dataset():
#     (X_train, y_train_raw), (X_test, y_test_raw) = mnist.load_data()
#     # normalize x
#     X_train = X_train / 255.0
#     X_test = X_test/ 255.0

#     # One hot encoding
#     # I.e. take the y value of 1 and turn in into [0,1,0,0,0...]
#     y_train=np.zeros((y_train_raw.shape[0],10))
#     for i in range(y_train.shape[0]):
#         y_train[i][y_train_raw[i]]=1
#     y_test=np.zeros((y_test_raw.shape[0],10))
#     for i in range(y_test.shape[0]):
#         y_test[i][y_test_raw[i]]=1
#     return X_train, y_train, X_test, y_test

# def load_data()
#     # LOAD DATA AND RESHAPE IT AND DO ALL THAT STUFF - model after Patrick's
#     # inp is for input and out is for output or what the number actually corresponds to




plotImages from keras tutorial that she got from tensorflow website
def plotImages(images_arr)
    fig, azes = plt.subplots(1, 20, figsize = (20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages (imgs)
print(labels)

# from keras tutorial
 def model() #don't know if this needs to be a function or not
    model = Sequential([                                                     #3 is for colour format but do these have colour?? Should be black and white
        Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'same', inpute_shape = (224,224,3)),
    #    review what this stuff means - other deep lizard videos
        MaxPool2d(pool_size = (2,2), strides = 2),
        Conv2D = 64, kernel_size = (3,3), activation = 'relu', padding = 'same'),
         MaxPool2d(pool_size = (2,2), strides = 2),
         Flatten(), #makes 1D I think
         Dense(units = 2, activation = 'softmax'),
    
    ])

    # model.summary() run this to see summary 
    #can use this since categroical_crossentropy works for more than 2 outputs
    model.compile(optimizer = Adam(learning_rate=0.0001), loss = 'categorical_crossentropy', metrics=['accuracy'])

    # make sure data names match this or change this
    # train_batches should have x and y data
    model.fit(x=train_batches, validation_data = valid_batches, epochs = 10, verbose = 2)

    #AFTER THIS HAS BEEN RAN CHECK ACCURACY TO CHECK FOR OVER OR UNDER FITTING

def predict() #don't know if this should be a function or not
    test_imgs, test_labels = next(test_batches)
    plotImages(test_imgs)
    print(test_labels)

    # will this work with one hot encoding?
    test_batches.classes

    predictions = model.predict(x=test_batches, verbose = 0)
    np.round(predictions)

    # confusion matrix checks if predictions match truth - I think should be 10x10
                                                            #argmax passes in index of most probable prediction
cm = confusion_matrix(y_true= test_batches.classes, y_pred = np.argmax(predictions, axis = -1))

# from scikit learn but from cat dog example so does it need to be changed to work for something with 10 possibilities
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    test_batches.classes_indices

    # this will plot the confusion matrix so I can see how well the model worked and where it went wrong
    cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plot_confusion_matrix(cm = cm, classes = cm_plot_labels, title = 'Confusion Matrix')



