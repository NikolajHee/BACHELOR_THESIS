


import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from code.scripts.data_backup import PTB_XL
import torch.nn.functional as F


def classifier_train(classifier_name, H, N_points, device):
    if classifier_name == 'svc':
        classifier = svm.SVC(kernel='rbf') 
    elif classifier_name == 'logistic':
        classifier = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga')

    dataset = PTB_XL(batch_size=N_points, shuffle_=True)


    X, y = dataset.load_some_signals()

    z = H(X.to(device))
    z = F.max_pool1d(z, kernel_size=z.shape[1])

    z = z.detach().cpu().numpy().reshape(z.shape[0], -1)
    y = y.numpy()

    classifier.fit(z, y)
        
    accuracy = np.mean(classifier.predict(z) == y)

    baseline = max(np.sum(y == 0), np.sum(y==1))/len(y)

    return accuracy, baseline

