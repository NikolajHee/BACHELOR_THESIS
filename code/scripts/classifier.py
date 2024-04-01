


import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression


from data_backup import PTB_XL



def classifier_train(classifier_name, H, N_points):
    if classifier_name == 'svc':
        classifier = svm.SVC(kernel='rbf') 
    elif classifier_name == 'logistic':
        classifier = LogisticRegression()

    dataset = PTB_XL(batch_size=N_points, shuffle_=True)


    X, y = dataset.load_some_signals()

    z = H(X)

    z = z.detach().numpy().reshape(z.shape[0], -1)
    y = y.numpy()

    classifier.fit(z, y)
        
    accuracy = classifier.predict(z) 

    return accuracy

