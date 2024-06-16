"""
Membership Inference Attack
- the model should attack based on ? and predict binary wether or not an observation is contained in the training data.
"""


import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression


class MIA_base():
    def __init__(self, device):
        self.device = device

    def predict(self, model, x, y):
        raise NotImplementedError()


    def train(self, model, x, y, contained):
        metric = self.metric(model, x, y)
        
        #thresholds = np.sort(np.unique(metric))

        self.classifier = LogisticRegression()

        self.classifier.fit(metric.reshape(-1,1), contained)

        # pred = np.zeros(len(thresholds))

        # for i, thres in enumerate(thresholds):
        #     pred[i] = np.mean(self.cond(thres, metric, y) == contained.astype(np.bool_))
        
        # self.threshold = thresholds[np.argmax(pred)]



    def evaluate(self, xs, ys, contains, model):
        predicted = self.predict(model, xs, ys)

        # return accuracy
        return np.mean(contains==predicted)


class PredictionCorrectnessMIA(MIA_base):
    def __init__(self, device):
        super().__init__(device)

    
    def train(self, model, x, y, contains):
        pass


    def predict(self, model, x, y):
        y_pred = model.predict_proba(x, self.device)

        y_pred = np.argmax(y_pred, axis=1)

        true_label =  (y_pred == y)

        # predict the correct classifications, to be in the training set
        return true_label
    
    


class PredictionConfidenceBasedMIA(MIA_base):
    def __init__(self, device):
        super().__init__(device)

    def metric(self, model, x, y):
        y_pred = model.predict_proba(x, self.device)

        return np.max(y_pred, axis=1)
    
    def cond(self, thres, metric, y):
        return metric >= thres


    def predict(self, model, x, y):
        confidence = self.metric(model, x, y)

        true_label =  self.classifier.predict(confidence.reshape(-1,1)) #confidence >= self.threshold

        # predict the correct classifications, to be in the training set
        return true_label
    


class PredictionEntropyBasedMIA(MIA_base):
    def __init__(self, device):
        super().__init__(device)
    
    def metric(self, model, x, y):
        y_pred = model.predict_proba(x, self.device)

        return -np.sum(y_pred * np.log(y_pred), axis=1)
    
    def cond(self, thres, metric, y):
        return metric <= thres



    def predict(self, model, x, y):
        entropy = self.metric(model, x, y)

        contain = self.classifier.predict(entropy.reshape(-1,1))

        #contain = entropy <= self.threshold

        return contain


class ModifiedPredictionEntropyBasedMIA(MIA_base):
    def __init__(self, device):
        super().__init__(device)

    
    def metric(self, model, x, y):
        y_pred = model.predict_proba(x, self.device).T

        mask = np.argmax(y_pred, axis=0) == y

        mask = mask.astype(np.bool_)

        mod_entropy = -np.sum((y_pred * np.log(1 - y_pred)) * (~mask) + 
                            (1 - y_pred) * np.log(y_pred) * (mask), axis=0)

        return mod_entropy
    
    def cond(self, thres, metric, y):
        return metric <= thres

    def predict(self, model, x, y):

        mod_entropy = self.metric(model, x, y)

        return self.classifier.predict(mod_entropy.reshape(-1,1)) #mod_entropy <= self.threshold

        



class MIA:
    def __init__(self, device):
        self.PredictionCorrectnessMIA = PredictionCorrectnessMIA(device)
        self.PredictionConfidenceBasedMIA = PredictionConfidenceBasedMIA(device)
        self.PredictionEntropyBasedMIA = PredictionEntropyBasedMIA(device)
        self.ModifiedPredictionEntropyBasedMIA = ModifiedPredictionEntropyBasedMIA(device)

        self.device = device
    
    def matrix(self, train):
        T, D = train[0][0].shape

        xs = np.zeros((len(train), T, D))
        ys = np.zeros((len(train)))

        contains = np.zeros((len(train)))


        for i, (x, y, contained) in enumerate(train):
            xs[i, :, :] = x.numpy()
            ys[i] = y[0]
            contains[i] = contained


        return xs, ys, contains


    def train(self, model, train_data):
        xs, ys, contains = self.matrix(train_data)

        train_accuracies = []

        for mia in [self.PredictionCorrectnessMIA, self.PredictionConfidenceBasedMIA, self.PredictionEntropyBasedMIA, self.ModifiedPredictionEntropyBasedMIA]:
            mia.train(model, xs, ys, contains)

            train_accuracies.append(mia.evaluate(xs, ys, contains, model))
        
        return train_accuracies

    def evaluate(self, model, data):

        xs, ys, contains = self.matrix(data)

        performance = []
        for mia in [self.PredictionCorrectnessMIA, self.PredictionConfidenceBasedMIA, self.PredictionEntropyBasedMIA, self.ModifiedPredictionEntropyBasedMIA]:
            performance.append(mia.evaluate(xs, ys, contains, model))
        
        return performance






class mia_loss:
    def __init__(self, device):
        self.device = device

        self.classifier = LogisticRegressionCV(cv=5)

    def train(self, model, train_data):
        
        loss_list = np.zeros(len(train_data))
        contained = np.zeros(len(train_data))

        for i, (x, y, contain) in enumerate(train_data):
            loss_list[i] = model.loss(x.to(self.device)[None,:,:], alpha=0.5)
            contained[i] = contain

        self.classifier.fit(loss_list.reshape(-1,1), contained)
        
    
    def evaluate(self, model, data):
        loss_list = np.zeros(len(data))
        contained = np.zeros(len(data))

        for i, (x, y, contain) in enumerate(data):
            loss_list[i] = model.loss(x.to(self.device)[None,:,:], alpha=0.5)
            contained[i] = contain

        return self.classifier.score(loss_list.reshape(-1,1), contained)
    

    def train_dp(self, model, train_data):

        self.classifiers = [LogisticRegression() for _ in range(len(model.models))]
        for i, model_ in enumerate(model.models):
            loss_list = np.zeros(len(train_data))
            contained = np.zeros(len(train_data))

            for j, (x, y, contain) in enumerate(train_data):
                loss_list[j] = model_.loss(x.to(self.device)[None,:,:], alpha=0.5)
                contained[j] = contain

            self.classifiers[i].fit(loss_list.reshape(-1,1), contained)
        
    
    def evaluate_dp(self, model, data):
        class_ = np.zeros((len(model.models), len(data)))
        for j, model_ in enumerate(model.models):
            loss_list = np.zeros(len(data))
            contained = np.zeros(len(data))

            for i, (x, y, contain) in enumerate(data):
                loss_list[i] = model_.loss(x.to(self.device)[None,:,:], alpha=0.5)
                contained[i] = contain

            class_[j,:] = self.classifiers[j].predict(loss_list.reshape(-1,1))

        from scipy.stats import mode
        votes = mode(class_, keepdims=False)[0]

        return np.mean(votes == contained)













