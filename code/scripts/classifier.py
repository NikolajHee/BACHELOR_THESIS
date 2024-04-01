


import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression


from code.scripts.data_backup import PTB_XL



def classifier_train(classifier_name, H, N_points, device):
    if classifier_name == 'svc':
        classifier = svm.SVC(kernel='rbf') 
    elif classifier_name == 'logistic':
        classifier = LogisticRegression()

    dataset = PTB_XL(batch_size=N_points, shuffle_=True)


    X, y = dataset.load_some_signals()

    z = H(X.to(device))

    z = z.detach().cpu().numpy().reshape(z.shape[0], -1)
    y = y.numpy()

    classifier.fit(z, y)
        
    accuracy = np.mean(classifier.predict(z) == y)

    return accuracy

