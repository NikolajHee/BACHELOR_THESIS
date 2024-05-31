"""
Membership Inference Attack
- the model should attack based on ? and predict binary wether or not an observation is contained in the training data.
"""


import numpy as np



class MIA():
    def __init__(self, ):
        pass

    def predict(self, model, x, y):
        raise NotImplementedError()


    def train(self, model, x, y):
        metric = self.metric(model, x)

        thresholds = np.sort(np.unique(metric))

        pred = np.zeros(len(thresholds))

        for i, thres in enumerate(thresholds):
            pred[i] = self.cond(thres, metric, y)

        return thresholds[np.argmax(pred)]


    def evaluate(self, dataset, model):
        
        contains = np.zeros(len(dataset))
        predicted = np.zeros(len(dataset))

        for i, (x, y, contained) in enumerate(dataset):
            predicted[i] = self.predict(model, x, y)
            contains[i] = contained

        # return accuracy
        return np.mean(contains==predicted)


class PredictionCorrectnessMIA(MIA):
    def __init__(self, ):
        pass

    
    def train(self, model, x, y):
        pass


    def predict(self, model, x, y):
        y_pred = model(x)

        true_label =  (y_pred == y)

        # predict the correct classifications, to be in the training set
        return true_label
    
    


class PredictionConfidenceBasedMIA(MIA):
    def __init__(self, threshold):
        self.threshold = threshold

    def metric(self, model, x, y):
        y_pred = model.predict_proba(x)

        return np.max(y_pred, axis=0)


    def predict(self, model, x, y):
        confidence = self.metric(model, x, y)

        true_label =  confidence >= self.threshold

        # predict the correct classifications, to be in the training set
        return true_label
    


class PredictionEntropyBasedMIA(MIA):
    def __init__(self, threshold):
        self.threshold = threshold
    
    def metric(self, model, x, y):
        y_pred = model.predict_proba(x)

        return -np.sum(y_pred * np.log(y_pred), axis=0)


    def predict(self, model, x, y):
        entropy = self.metric(model, x, y)

        contain = entropy <= self.threshold

        return contain


class ModifiedPredictionEntropyBasedMIA(MIA):
    def __init__(self, threshold):
        self.threshold = threshold

    
    def metric(self, model, x, y):
        y_pred = model.predict_proba(x)

        mask = 1 - y

        mod_entropy = -np.sum((y_pred * np.log(1 - y_pred)) * (mask) + 
                            (1 - y_pred) * np.log(y_pred) * (~mask), axis=0)

        return mod_entropy

    def predict(self, model, x, y):

        mod_entropy = self.metric(model, x, y)

        return mod_entropy <= self.threshold

        





def metric_based_mia():

    PredictionCorrectnessMIA()

    PredictionConfidenceBasedMIA()


if __name__ == '__main__':

    PredictionEntropyBasedMIA()



