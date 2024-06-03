"""
Membership Inference Attack
- the model should attack based on ? and predict binary wether or not an observation is contained in the training data.
"""


import numpy as np



class MIA_base():
    def __init__(self, device):
        self.device = device

    def predict(self, model, x, y):
        raise NotImplementedError()


    def train(self, model, x, y, contained):
        metric = self.metric(model, x, y)

        thresholds = np.sort(np.unique(metric))

        pred = np.zeros(len(thresholds))

        for i, thres in enumerate(thresholds):
            pred[i] = np.mean(self.cond(thres, metric, y) == contained.astype(np.bool_))
        
        self.threshold = thresholds[np.argmax(pred)]



    def evaluate(self, dataset, model):
        
        contains = np.zeros(len(dataset))
        predicted = np.zeros(len(dataset))

        for i, (x, y, _, contained) in enumerate(dataset):
            z = model.encode(x.to(self.device)[None,:,:])

            predicted[i] = self.predict(model, z, y)

            contains[i] = contained

        # return accuracy
        return np.mean(contains==predicted)


class PredictionCorrectnessMIA(MIA_base):
    def __init__(self, device):
        super().__init__(device)

    
    def train(self, model, x, y, contains):
        pass


    def predict(self, model, x, y):
        y_pred = model.predict_proba(x)

        y_pred = np.argmax(y_pred, axis=1)
        
        true_label =  (y_pred == y.item())

        # predict the correct classifications, to be in the training set
        return true_label
    
    


class PredictionConfidenceBasedMIA(MIA_base):
    def __init__(self, device):
        super().__init__(device)

    def metric(self, model, x, y):
        y_pred = model.predict_proba(x)

        return np.max(y_pred, axis=1)
    
    def cond(self, thres, metric, y):
        return metric >= thres


    def predict(self, model, x, y):
        confidence = self.metric(model, x, y)

        true_label =  confidence >= self.threshold

        # predict the correct classifications, to be in the training set
        return true_label
    


class PredictionEntropyBasedMIA(MIA_base):
    def __init__(self, device):
        super().__init__(device)
    
    def metric(self, model, x, y):
        y_pred = model.predict_proba(x)

        return -np.sum(y_pred * np.log(y_pred), axis=1)
    
    def cond(self, thres, metric, y):
        return metric <= thres



    def predict(self, model, x, y):
        entropy = self.metric(model, x, y)

        contain = entropy <= self.threshold

        return contain


class ModifiedPredictionEntropyBasedMIA(MIA_base):
    def __init__(self, device):
        super().__init__(device)

    
    def metric(self, model, x, y):
        y_pred = model.predict_proba(x).T

        mask = (1 - np.array(y)).astype(np.bool_)

        mod_entropy = -np.sum((y_pred * np.log(1 - y_pred)) * (mask) + 
                            (1 - y_pred) * np.log(y_pred) * (~mask), axis=0)

        return mod_entropy
    
    def cond(self, thres, metric, y):
        return metric <= thres

    def predict(self, model, x, y):

        mod_entropy = self.metric(model, x, y)

        return mod_entropy <= self.threshold

        

class MIA:
    def __init__(self, device):
        self.PredictionCorrectnessMIA = PredictionCorrectnessMIA(device)
        self.PredictionConfidenceBasedMIA = PredictionConfidenceBasedMIA(device)
        self.PredictionEntropyBasedMIA = PredictionEntropyBasedMIA(device)
        self.ModifiedPredictionEntropyBasedMIA = ModifiedPredictionEntropyBasedMIA(device)

        self.device = device
    
    def matrix(self, model, train):
        zs = np.zeros((len(train), model.output_dim))

        ys = np.zeros((len(train)))

        contains = np.zeros((len(train)))

        for i, (x, y, _, contained) in enumerate(train):
            zs[i, :] = model.encode(x.to(self.device)[None, :, :])
            ys[i] = y
            contains[i] = contained

        return zs, ys, contains


    def train(self, model, train_data):
        zs, ys, contains = self.matrix(model, train_data)

        train_accuracies = []

        for mia in [self.PredictionCorrectnessMIA, self.PredictionConfidenceBasedMIA, self.PredictionEntropyBasedMIA, self.ModifiedPredictionEntropyBasedMIA]:
            mia.train(model, zs, ys, contains)

            train_accuracies.append(mia.evaluate(train_data, model))
        
        return train_accuracies

    def evaluate(self, model, data):
        performance = []
        for mia in [self.PredictionCorrectnessMIA, self.PredictionConfidenceBasedMIA, self.PredictionEntropyBasedMIA, self.ModifiedPredictionEntropyBasedMIA]:
            performance.append(mia.evaluate(data, model))
        
        return performance








class MIEncoder(MIA_base):
    def __init__(self, ):
        pass

    def train(self, model, x, y, contained):
        pass

    










