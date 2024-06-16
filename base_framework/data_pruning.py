"""
-------------
Data Pruning
-------------

- This script contains the method for splitting the data into shards and shards into slices,
whereafter the model described in the thesis is trained on each shard.
- is run in unlearn.py



* Slice
    inheriting functionality from Subset. Adds the original data_indices

* ShardsAndSlices
    The class for splitting data into shards and slices.

* Pruning
    The main Pruning class

"""

# imports
from torch.utils.data import Subset
import numpy as np
import torch
import random
from scipy.stats import mode
import torch.nn.functional as F

# own imports
from base_framework.utils import save_parameters, random_seed, TimeTaking
from base_framework.dataset import *




class Slice(Subset):
    """
    Class to keep the indices from the original dataset.
    """
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)

        if (type(dataset.dataset) is PTB_XL) or (type(dataset.dataset) is AEON_DATA):
            self.data_indices = dataset.indices[indices]
        elif (type(dataset.dataset.dataset) is PTB_XL) or ((type(dataset.dataset.dataset) is AEON_DATA)):
            self.data_indices = dataset.dataset.indices[dataset.indices[indices]]
        

    
    def __contains__(self, index):
        return index in self.data_indices


class ShardsAndSlices:
    """
    Shards and Slices the dataset into N_shards and N_slices
    """
    def __init__(self, dataset, N_shards:int, N_slices:int, seed = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.unlearned_points = None

        self.N_shards, self.N_slices = N_shards, N_slices

        # assign each point to a shard
        indices_shards = [i for i in range(N_shards) for j in range(len(dataset)//N_shards)]

        # add index for each indice of the rest
        if len(dataset) % self.N_shards > 0:
            indices_shards += [i for i in range(len(dataset) % N_shards)]

        # shuffle
        random.shuffle(indices_shards)

        indices_shards = np.array(indices_shards)


        self.shards = []

        self.slices = [[] for _ in range(N_shards)]

        for i in range(N_shards):
            # get the adjoint indices
            index_shards = np.where(indices_shards==i)[0]

            self.shards.append(Slice(dataset, index_shards))

            # assign each point to a slice
            indices_slices = [j for j in range(N_slices) for _ in range(len(self.shards[i])//N_slices)]

            # add index for each indice of the rest
            if len(self.shards[i]) % N_slices > 0:
                indices_slices += [j for j in range(len(self.shards[i]) % N_slices)]

            # shuffle
            random.shuffle(indices_slices)

            indices_slices = np.array(indices_slices)

            for j in range(N_slices):
                # get overlapping indices
                index_slices = np.where(indices_slices <= j)[0]

                self.slices[i].append(Slice(self.shards[i], index_slices))
        
    def contains(self, x):
        """
        Return which shard and slice contains the index x
        """
        shard_contain = [x in shard for shard in self.shards]

        assert (not all([not i for i in shard_contain])), \
                f"{x} is not in any shard"
        

        # shard_index contains the shard on which the model should be retrained
        shard_index = np.argmax(shard_contain)

        #print(shard_contain, shard_index)



        #print([slice_.data_indices for slice_ in self.slices[shard_index]])
        #print(x)

        slice_contain = [x in slice_ for slice_ in self.slices[shard_index]]

        assert (all([not i for i in slice_contain]) == False)

        slice_index = np.argmax(np.array(slice_contain))

        return shard_index, slice_index
    
    def remove_points(self, x, indices):
        def get_mask(indices, remove_indices):
            temp = indices
            mask = np.zeros_like(temp)
            for index in remove_indices:
                mask += temp == index
            
            return ~mask.astype(bool)
        
        for shard_index, slice_index in enumerate(x):
            print(f"removing {shard_index} {slice_index}")
            if len(slice_index) > 0:
                for j in np.unique(slice_index):
                    ind_ = self.slices[shard_index][j].data_indices
                    mask = get_mask(ind_, indices)

                    self.slices[shard_index][j].data_indices = self.slices[shard_index][j].data_indices[mask]
                    self.slices[shard_index][j].indices = self.slices[shard_index][j].indices[mask]
        
        # save unlearned points in new dataset
        if self.unlearned_points is None:
            unlearned_indices = [np.argmax(i == self.shards[0].dataset.indices) for i in indices]
            self.unlearned_points = Subset(self.shards[0].dataset, unlearned_indices)
            




def return_classifier(classifier_name:str):
    """
    Return the classifier based on the classifier_name
    """
    if classifier_name == 'svc':
        from sklearn import svm
        return svm.SVC(kernel='rbf') 

    elif classifier_name == 'logistic':
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    elif classifier_name == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=5)

    else:
        raise ValueError(f'{classifier_name} is not a choice')

from base_framework.TS2VEC import TS2VEC




class Pruning:
    def __init__(self, 
                 dataset, 
                 N_shards:int, 
                 N_slices:int, 
                 input_dim:int,
                 hidden_dim:int,
                 output_dim:int,
                 p:float,
                 device,
                 classifier_name:str,
                 seed=None):
        """
        The main Data Pruning class.

        Contains:
        - encode
        - evaluate
        - train_classifiers
        - evaluate_classifiers
        - collect_matrix
        - collect_matrix2
        - train_temp
        - train
        - unlearn
        - forward
        - predict_proba
        - load
        """
        
        self.data = ShardsAndSlices(dataset=dataset,
                                    N_shards=N_shards,
                                    N_slices=N_slices,
                                    seed=seed)
        self.models = []
        self.classifiers = []

        self.device = device

        self.model_settings = {'input_dim': input_dim,
                               'hidden_dim': hidden_dim,
                               'output_dim': output_dim,
                               'p':p,
                               'device':device}
        
        self.output_dim = output_dim

        for i in range(N_shards):
            model = TS2VEC(**self.model_settings)
    
            classifier = return_classifier(classifier_name)
        
            self.models.append(model)

            self.classifiers.append(classifier)
        
    

    def encode(self,
               model,
               X):
        
            # encode the data
            if type(X) is np.ndarray: X = torch.tensor(X)
  
            z = model.model(X.float().to(self.device))

            # maxpool over time dimension
            z = z.transpose(1,2)

            z = F.max_pool1d(z, kernel_size=z.shape[2])

            z = z.detach().cpu().numpy().reshape(z.shape[0], -1)

            return z



    def evaluate(self, data):
        i = 0
        ind_accuracy = []

        predictions = np.zeros((len(self.classifiers), len(data)))
        for model, classifier in zip(self.models, self.classifiers):
            Z, Y = self.collect_matrix2(model, data)

            predictions[i, :] = classifier.predict(Z)

            ind_accuracy.append(np.mean(classifier.predict(Z) == Y.squeeze()))

            i+=1
        
        # test majority voting
        votes = mode(predictions, keepdims=False)[0]

        accuracy = np.mean(Y.squeeze() == votes.ravel())

        return accuracy, ind_accuracy
    

    def train_classifiers(self):
        ind_train_accuracy = []
        

        i = 0
        for model, classifier in zip(self.models, self.classifiers):
            Z_train, Y_train = self.collect_matrix2(model, self.data.shards[i])

            classifier.fit(Z_train, Y_train.squeeze())

            ind_train_accuracy.append(np.mean(classifier.predict(Z_train) == Y_train.squeeze()))

            i+=1

        
        return ind_train_accuracy


    def evaluate_classifiers(self, test_dataset):
        
        unlearn_accuracy = None


        if self.data.unlearned_points is not None:
            unlearn_predictions = np.zeros((len(self.classifiers), len(self.data.unlearned_points)))


        i = 0
        test_predictions = np.zeros((len(self.classifiers), len(test_dataset)))
        ind_train_accuracy = []

        for model, classifier in zip(self.models, self.classifiers):
            Z_train, Y_train = self.collect_matrix2(model, self.data.shards[i])
            Z_test, Y_test = self.collect_matrix2(model, test_dataset)

            classifier.fit(Z_train, Y_train.squeeze())


            test_predictions[i, :] = classifier.predict(Z_test)

            ind_train_accuracy.append(np.mean(classifier.predict(Z_train) == Y_train.squeeze()))
  

            
            if self.data.unlearned_points is not None:
                Z_unlearn, Y_unlearn = self.collect_matrix2(model, self.data.unlearned_points)

                unlearn_predictions[i, :] = classifier.predict(Z_unlearn)

            i+=1
        


        # test majority voting
        votes = mode(test_predictions, keepdims=False)[0]

        test_accuracy = np.mean(Y_test.squeeze() == votes.ravel())



        ind_test_accuracy = np.mean(test_predictions == np.repeat(Y_test, self.data.N_shards, axis=1).T, axis=1)

        # unlearn majority voting
        if self.data.unlearned_points is not None:
            votes = mode(unlearn_predictions, keepdims=False)[0]

            unlearn_accuracy = np.mean(Y_unlearn.squeeze() == votes.ravel())

        return ind_test_accuracy, ind_train_accuracy, test_accuracy, unlearn_accuracy



    def collect_matrix(self, dataset):
        N = len(dataset)
        T, d = dataset[0][0].shape

        output_dim = dataset[0][1].shape if dataset[0][1].shape else 1
            
        X = torch.zeros((N, T, d))
        y = torch.zeros((N, output_dim))
        for i in range(len(dataset)):
            X[i], y[i, :] = dataset[i]

        return X, y
    
    def collect_matrix2(self, model, dataset):
        N = len(dataset)
        z_dim = model.output_dim
        T, d = dataset[0][0].shape

        output_dim = dataset[0][1][0].shape if dataset[0][1][0].shape else 1
            
        Z = np.zeros((N, z_dim))
        y = np.zeros((N, output_dim))
        for i in range(len(dataset)):
            X, y[i, :] = dataset[i][0], dataset[i][1][0].numpy()
            Z[i] = self.encode(model, X[None,:,:])

        return Z, y
    

    def train_temp(self,
                   shard_index:int,
                   slice_index:int,
                   n_epochs:int,
                   batch_size:int,
                   learning_rate:float,
                   grad_clip,
                   alpha:float,
                   wandb,
                   path=None):
        
        if path:
            self.models[shard_index].model.load_state_dict(torch.load(path))
        else:
            self.models[shard_index] = TS2VEC(**self.model_settings)

        for j in range(slice_index, self.data.N_slices):
            print(j)

            self.models[shard_index].temp(dataset=self.data.slices[shard_index][j],
                                          n_epochs=n_epochs[j],
                                          batch_size=batch_size,
                                          learning_rate=learning_rate,
                                          grad_clip=grad_clip,
                                          alpha=alpha,
                                          wandb=wandb,
                                          train_path=None)
                

    def train(self, 
              n_epochs:list[int],
              batch_size:int,
              learning_rate:float,
              grad_clip:float,
              alpha:float,
              wandb,
              save_path:str,
              time_taking: TimeTaking,
              test_dataset=None):
        """
        Train the model sequentially.   
        Afterwards train the classifier on the features
        """

        assert len(n_epochs) == self.data.N_slices, \
                f"amount of epochs ({len(n_epochs)}) differs "  \
                f"from the amount of slices ({self.data.N_slices})"


        test_accuracy = np.zeros(((self.data.N_slices)))

        time_taking.start('Overall Training')

        for j in range(self.data.N_slices):

            print(f"Training models for {n_epochs[j]} epochs.")
            
            losses = []

            for i, model in enumerate(self.models):

                loss = model.temp(dataset=self.data.slices[i][j],
                                n_epochs=n_epochs[j],
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                grad_clip=grad_clip,
                                alpha=alpha,
                                wandb=wandb,
                                train_path=save_path)

                
                torch.save(model.model.state_dict(), 
                           os.path.join(save_path, f'shard_{i}', f'slice_{j}', 'model.pt'))
                
                losses.append(loss)
                
            save = {f"loss_model{i}": np.mean(x) for i, x in enumerate(losses)}


            time_taking.pause('Overall Training')

            time_save = time_taking.output_dict('Overall Training')

            save.update(time_save)

            #wandb.log(save)

            ind_train_accuracy, ind_test_accuracy, test_accuracy[j], _ = self.evaluate_classifiers(test_dataset=test_dataset)

            save.update({'test_accuracy': test_accuracy[j]})

            save.update({f"train_acc_model{i}": ind_train_accuracy[i] for i in range(len(ind_train_accuracy))})

            save.update({f"test_acc_model{i}": ind_test_accuracy[i] for i in range(len(ind_test_accuracy))})

            print(f"temp results: {save}")
            wandb.log(save)
            
        time_taking.end('Overall Training')

        time_taking.save()
                

        return test_accuracy
    


    def unlearn(self, 
                indices,
                n_epochs:list[int],
                batch_size:int,
                learning_rate:float,
                grad_clip:float,
                alpha:float,
                wandb,
                save_path:str,
                time_taking: TimeTaking,
                test_dataset):

        time_taking.start('Overall Unlearning')

        save = [[] for _ in range(len(self.models))]

        models_to_be_updated = set()
 
        for index in indices:
            shard_index, slice_index = self.data.contains(index)

            save[shard_index].append(slice_index)

            models_to_be_updated.add(shard_index)

            #model_to_update.add(shard_index)

        print(f'Updating models {models_to_be_updated}.')
        #print(save)
        # remove all the points which should be unlearned
        #print(f"removing {indices}")
        self.data.remove_points(save, indices)

        # train the models that needs to be retrained
        for (shard_index, slice_index) in enumerate(save):
            if len(slice_index) > 0:
                
                # train from slice_
                slice_ = min(slice_index)

                # train function that should, given a shard index 
                # and a slice index, train the given shard, 
                # from the slice and forward.

                # if the min-slice is above 1, the path should be given

                #print(f'Starting train, shard: {shard_index}, slice: {slice_index}.')
                self.train_temp(shard_index=shard_index,
                                slice_index=min(slice_index),
                                n_epochs=n_epochs,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                grad_clip=grad_clip,
                                alpha=alpha,
                                wandb=wandb,
                                path=os.path.join(save_path, f'shard_{shard_index}', f'slice_{slice_-1}', 'model.pt') if slice_>0 else None)
                

        time_taking.end('Overall Unlearning')
        time_ = time_taking.output_dict('Overall Unlearning')
        
        return time_

           


    def forward(self, x):
        """
        Forward pass through the model
            for each shard
            and
            aggregation
        """
        x = x.float()
        if x.dim() == 2:
            T, d = x.shape
            x = x.view(1, T, d)

        y = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            z = model.model(x)


        
        unique, counts = np.unique(y, return_counts=True)

        return unique[np.argmax(counts)]
    
    def predict_proba(self, x, device):

        if len(x.shape) == 2:
            x = x[None,:,:]
            N = 1
        else:
            N = x.shape[0]

        prob = np.zeros((N,len(self.classifiers[0].classes_)))

        for (model, classifier) in zip(self.models, self.classifiers):
            # z = model.model.forward(x.to(device).float()[None,:,:])
            
            # z = z.transpose(1,2)
            # z = F.max_pool1d(z, kernel_size=z.shape[2])
            # z = z.detach().cpu().numpy().reshape(z.shape[0], -1)

            z = self.encode(model, x)

            #print(z.shape)
            #print(classifier.predict_proba(z).shape)

            prob += classifier.predict_proba(z)
        
        return prob/len(self.models)
    
    def load(self, 
             save_path,
             device):
        """
        load model given path and device.
        """
        
        
        for i in range(self.data.N_shards):
            self.models[i].model.load_state_dict(torch.load(os.path.join(save_path, f'shard_{i}/slice_{self.data.N_slices-1}' + '/model.pt'), map_location=device))
    
    def loss(self, x, alpha):

        loss_list = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            loss_list[i] = model.loss(x, alpha=alpha)

        return np.mean(loss_list)
        


if __name__ == '__main__':
    def folder_structure(save_path, N_shards, N_slices):
        save_path = os.path.join(save_path, 'model_save')
        for i in range(N_shards):
            for j in range(N_slices):

                os.makedirs(os.path.join(save_path, f"shard_{i}/slice_{j}"), exist_ok=True)
        
        return save_path

    train_size = 60
    test_size = 100
    test_proportion = 0.3
    N_shards = 2
    N_slices = 2
    input_dim = 12
    hidden_dim = 32
    output_dim = 320
    p = 0.5
    batch_size = 5
    epochs = [1]*N_slices
    learning_rate = 0.001
    save_path = ''

    print(f"1.7MB per model. Therefore for {N_shards*N_slices} models, it needs {N_shards*N_slices*1.6628:.2f} MB")

    assert N_shards*N_slices*1.6628 < 1000, \
            f"The model will use above 1000MB in space! " \
            f"It will use around {N_shards*N_slices*1.6628} MB."

    save_path = folder_structure(save_path=save_path, N_shards=N_shards, N_slices=N_slices)

    from base_framework.utils import random_seed

    time = TimeTaking(save_path=save_path, verbose=False)

    random_seed(1)
    dataset = PTB_XL('/zhome/51/4/167677/Desktop/BACHELOR_THESIS/PTB_XL')

    from base_framework.utils import train_test_dataset
    train_dataset, test_dataset, D = train_test_dataset(dataset=dataset,
                                                     test_proportion=test_proportion,
                                                     train_size=train_size,
                                                     test_size=test_size,
                                                     return_stand=False)


    test = ShardsAndSlices(train_dataset, 10, 4)

    test.contains(train_dataset.indices[0])

    # test2 = Pruning(dataset=train_dataset, 
    #                 N_shards=N_shards, 
    #                 N_slices=N_slices, 
    #                 input_dim=input_dim, 
    #                 hidden_dim=hidden_dim, 
    #                 output_dim=output_dim, 
    #                 p=p, 
    #                 device='cpu',
    #                 classifier_name='logistic')

    
    # acc = test2.train(n_epochs=epochs, 
    #             batch_size=batch_size, 
    #             learning_rate=learning_rate, 
    #             grad_clip=None, 
    #             alpha=0.5, 
    #             wandb=None, 
    #             save_path=save_path, 
    #             time_taking=time,
    #             test_dataset=test_dataset)
    
    # print('-'*20)
    # print("UNLEARNING")
    # print('-'*20)
    # print(acc)

    # test2.unlearn(indices=train_dataset.indices[0:3],
    #               n_epochs=epochs,
    #               batch_size=batch_size,
    #               learning_rate=learning_rate,
    #               grad_clip=None,
    #               alpha=0.5,
    #               wandb=None,
    #               save_path=save_path,
    #               time_taking=time,
    #               test_dataset=test_dataset
    #             )

    # time.save()
