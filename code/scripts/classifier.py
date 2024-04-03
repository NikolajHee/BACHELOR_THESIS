


import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F
from tqdm import tqdm


def classifier_train(classifier_name, 
                    H, 
                    train_loader,
                    test_loader,
                    device,
                    num_batches=(tuple)):
    

    train_batches, test_batches = num_batches


    if classifier_name == 'svc':
        classifier = svm.SVC(kernel='rbf') 
    elif classifier_name == 'logistic':
        classifier = LogisticRegression()
    else:
        raise ValueError(f'{classifier_name} is not a choice')

    batch_size = train_loader.batch_size


    Z_train = np.zeros((batch_size * train_batches, H.output_dim))
    Y_train = np.zeros((batch_size * train_batches))
    Z_test= np.zeros((batch_size * test_batches, H.output_dim))
    Y_test = np.zeros(batch_size * test_batches)

    

    for i, (X, y) in tqdm(enumerate(train_loader)):
        z = H(X.to(device).float())

        # Maxpooling is inspried by the TS2VEC framework for classification
        #   maxpooling over time instances!
        z = F.max_pool1d(z, kernel_size=z.shape[2])

        z = z.detach().cpu().numpy().reshape(z.shape[0], -1)

        y = y.numpy()

        if i < len(train_loader) - 1:
            Z_train[i*batch_size:(i+1)*batch_size] = z.reshape(batch_size, H.output_dim)
            Y_train[i*batch_size:(i+1)*batch_size] = y.reshape(batch_size)
        # the last batch can be smaller
        else:
            Z_train[i*batch_size:] = z.reshape(-1, H.output_dim)
            Y_train[i*batch_size:] = y.reshape(-1)

        if (i+1) == train_batches: break
        


        


    classifier.fit(Z_train, Y_train)

    train_accuracy = np.mean(classifier.predict(Z_train) == Y_train)


    for i, (X, y) in tqdm(enumerate(test_loader)):
        z = H(X.to(device).float())

        # Maxpooling is inspried by the TS2VEC framework for classification
        z = F.max_pool1d(z, kernel_size=z.shape[2])

        z = z.detach().cpu().numpy().reshape(z.shape[0], -1)
        y = y.numpy()

        if i < len(test_loader) - 1:
            Z_test[i*batch_size:(i+1)*batch_size] = z.reshape(batch_size, H.output_dim)
            Y_test[i*batch_size:(i+1)*batch_size] = y.reshape(batch_size)
        # the last batch can be smaller
        else:
            Z_test[i*batch_size:] = z.reshape(-1, H.output_dim)
            Y_test[i*batch_size:] = y.reshape(-1)
        
        if (i+1) == test_batches: break

    #classifier.fit(Z_test, Y_test)

    test_accuracy = np.mean(classifier.predict(Z_test) == Y_test)


    return train_accuracy, test_accuracy



