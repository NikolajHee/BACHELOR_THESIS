"""
Classifier.py
    - This script is used to train a chosen classifier on the extracted features from the model.
    - The features are extracted from the model by using the forward pass and then maxpooling over the time dimension.
"""


# imports 
import numpy as np
from sklearn import svm
import torch.nn.functional as F
from utils import baseline
from torch.utils.data import DataLoader


def classifier_train(classifier_name:str, 
                     model, 
                     train_loader:DataLoader,
                     test_loader:DataLoader,
                     device):
    """
    Classifier training function.

        classifier_name : name of the classifier to use.
            - 'svc' : Support Vector Classifier
            - 'logistic' : Logistic Regression
            - 'knn' : K-Nearest Neighbors
        model : the model which can be used to extract features
        train_loader : the training dataloader
        test_loader : the test dataloader
        device : the device where the model resides
    """



    # get baseline accuracy (based on majority class)
    baseline_accuracy = baseline(train_dataset=train_loader.dataset, 
                                 test_dataset=test_loader.dataset)
    
    
    # choose classifier
    if classifier_name == 'svc':
        classifier = svm.SVC(kernel='rbf') 

    elif classifier_name == 'logistic':
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    elif classifier_name == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=5)

    else:
        raise ValueError(f'{classifier_name} is not a choice')


    # get output dimension and batch size
    output_dim = model.module.output_dim
    batch_size = train_loader.batch_size
    train_batches, test_batches = len(train_loader), len(test_loader)

    # initialize
    Z_train = np.zeros((batch_size * train_batches, output_dim))
    Y_train = np.zeros((batch_size * train_batches))
    Z_test= np.zeros((batch_size * test_batches, output_dim))
    Y_test = np.zeros(batch_size * test_batches)

    # train loop to convert all data to features
    for i, (X, y) in enumerate(train_loader):
        z = model(X.to(device).float()) # output: N x T x Dr
        
        
        z = z.transpose(1,2) # N x Dr x T


        # Maxpooling is inspried by the TS2VEC framework for classification
        #   maxpooling over time instances!
        z = F.max_pool1d(z, kernel_size=z.shape[2]) # N x Dr

        z = z.detach().cpu().numpy().reshape(z.shape[0], -1)

        y = y.numpy()

        Z_train[i*batch_size:(i+1)*batch_size] = z.reshape(batch_size, output_dim)
        Y_train[i*batch_size:(i+1)*batch_size] = y.reshape(batch_size)

    # fit the classifier to the features (based on the training data)
    classifier.fit(Z_train, Y_train)

    # calculate the accuracy on the training data
    train_accuracy = np.mean(classifier.predict(Z_train) == Y_train)

    # test loop to convert all data to features
    for i, (X, y) in enumerate(test_loader):
        z = model(X.to(device).float())

        z = z.transpose(1,2) # N x Dr x T

        # Maxpooling is inspried by the TS2VEC framework for classification
        #   maxpooling over time instances!
        z = F.max_pool1d(z, kernel_size=z.shape[2])

        z = z.detach().cpu().numpy().reshape(z.shape[0], -1)
        y = y.numpy()

        Z_test[i*batch_size:(i+1)*batch_size] = z.reshape(batch_size, output_dim)
        Y_test[i*batch_size:(i+1)*batch_size] = y.reshape(batch_size)

    # calculate the accuracy on the test data
    test_accuracy = np.mean(classifier.predict(Z_test) == Y_test)

    return train_accuracy, test_accuracy, baseline_accuracy



