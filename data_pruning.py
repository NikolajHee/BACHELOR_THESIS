"""
Data Pruning
"""

# imports
from torch.utils.data import Subset
import numpy as np
import torch
import random
from scipy.stats import mode
import torch.nn.functional as F

# own imports
from utils import save_parameters, random_seed, TimeTaking
from dataset import *




class Slice(Subset):
    """
    Class to keep the indices from the original dataset.
    """
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)

        if type(dataset.dataset) is PTB_XL:
            self.data_indices = dataset.indices[indices]
        elif type(dataset.dataset.dataset) is PTB_XL:
            self.data_indices = dataset.dataset.indices[dataset.indices[indices]]
        

    
    def __contains__(self, index):
        return index in self.data_indices


class ShardsAndSlices:
    """
    Shards and Slices the dataset into N_shards and N_slices
    """
    def __init__(self, dataset, N_shards, N_slices, seed = None):
        if seed is not None:
            np.random.seed(seed)
        
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

        # shard_index contains the shard on which the model should be retrained
        shard_index = np.argmax(shard_contain)

        slice_contain = [x in slice_ for slice_ in self.slices[shard_index]]
        slice_index = np.sum(~np.array(slice_contain))

        return shard_index, slice_index
    
    def remove_points(self, x, indices):
        def get_mask(indices, remove_indices):
            temp = indices
            mask = np.zeros_like(temp)
            for index in remove_indices:
                mask += temp == index
            
            return ~mask.astype(bool)
        
        for shard_index, slice_index in enumerate(x):
            if len(slice_index) > 0:
                for j in np.unique(slice_index):
                    ind_ = self.slices[shard_index][j].data_indices
                    mask = get_mask(ind_, indices)

                    self.slices[shard_index][j].data_indices = self.slices[shard_index][j].data_indices[mask]
                    self.slices[shard_index][j].indices = self.slices[shard_index][j].indices[mask]
        
        # save unlearned points in new dataset
        if  self.unlearned_points is None:
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

from TS2VEC import TS2VEC




class Pruning:
    def __init__(self, 
                 dataset, 
                 N_shards, 
                 N_slices, 
                 input_dim,
                 hidden_dim,
                 output_dim,
                 p,
                 device,
                 classifier_name:str,
                 seed=None):

        
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

        for i in range(N_shards):
            model = TS2VEC(**self.model_settings)
            
            classifier = return_classifier(classifier_name)
            
            self.models.append(model)

            self.classifiers.append(classifier)
        
    

    def encode(self,
               model,
               X):
        
            # encode the data
            z = model.model(X.float().to(self.device))

            # maxpool over time dimension
            z = z.transpose(1,2)

            z = F.max_pool1d(z, kernel_size=z.shape[2])

            z = z.detach().cpu().numpy().reshape(z.shape[0], -1)

            return z

    def evaluate_classifiers(self, test_dataset):
        
        unlearn_accuracy = None

        X_test, Y_test = self.collect_matrix(test_dataset)

        N_test, _ = Y_test.shape

        test_predictions = np.zeros((len(self.classifiers), N_test))

        if self.data.unlearned_points is not None:
            X_unlearn, Y_unlearn = self.collect_matrix(self.data.unlearned_points)
            N_unlearn, _ = Y_unlearn.shape
            unlearn_predictions = np.zeros((len(self.classifiers), N_unlearn))


        i = 0
        for model, classifier in zip(self.models, self.classifiers):
            Z_train, Y_train = self.collect_matrix2(model, self.data.shards[i])

            #Z_train = self.encode(model, X_train) #TODO : maybe move into TS2Vec

            classifier.fit(Z_train, Y_train.squeeze())


            Z_test = self.encode(model, X_test)

            test_predictions[i, :] = classifier.predict(Z_test)

            if self.data.unlearned_points is not None:
                Z_unlearn = self.encode(model, X_unlearn)

                unlearn_predictions[i, :] = classifier.predict(Z_unlearn)

            i+=1
        
        
        # majority voting
        votes = mode(test_predictions, keepdims=False)[0]

        accuracy = np.mean(Y_test.squeeze().numpy() == votes.ravel())

        # majority voting
        if self.data.unlearned_points is not None:
            votes = mode(unlearn_predictions, keepdims=False)[0]

            unlearn_accuracy = np.mean(Y_unlearn.squeeze().numpy() == votes.ravel())

        return accuracy, unlearn_accuracy



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

        output_dim = dataset[0][1].shape if dataset[0][1].shape else 1
            
        Z = torch.zeros((N, z_dim))
        y = torch.zeros((N, output_dim))
        for i in range(len(dataset)):
            X, y[i, :] = dataset[i]
            Z[i] = self.encode(model, X.view(1,T,d))

        return Z, y


    def train_temp(self,
                   shard_index,
                   slice_index,
                   n_epochs,
                   batch_size,
                   learning_rate,
                   grad_clip,
                   alpha,
                   wandb,
                   path=None):
        
        if path:
            self.models[shard_index].model.load_state_dict(torch.load(path))
        else:
            self.models[shard_index] = TS2VEC(**self.model_settings)

        for j in range(slice_index, self.data.N_slices):
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

        for j in range(self.data.N_slices):

            print(f"Training model {j}.")
            

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
                
                wandb.log({f'loss_model{i}': np.mean(loss)})

            test_accuracy[j], _ = self.evaluate_classifiers(test_dataset=test_dataset)
            if wandb is not None: 
                wandb.log({'test_accuracy': test_accuracy[j]})
                

            
    
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

        save = [[] for _ in range(len(self.models))]

        models_to_be_updated = set()
 
        for index in indices:
            shard_index, slice_index = self.data.contains(index)

            save[shard_index].append(slice_index)

            models_to_be_updated.add(shard_index)

            #model_to_update.add(shard_index)

        print(f'Updating models {models_to_be_updated}.')
        print(save)
        # remove all the points which should be unlearned
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


                self.train_temp(shard_index=shard_index,
                                slice_index=min(slice_index),
                                n_epochs=n_epochs,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                grad_clip=grad_clip,
                                alpha=alpha,
                                wandb=wandb,
                                path=os.path.join(save_path, f'shard_{shard_index}', f'slice_{slice_-1}', 'model.pt') if slice_>0 else None)
                
        test_accuracy, unlearn_accuracy = self.evaluate_classifiers(test_dataset=test_dataset)

        print(f"Test accuracy {test_accuracy}. Unlearn accuracy {unlearn_accuracy}.")

        if wandb is not None:
            wandb.log({'unlearn/test_accuracy': test_accuracy, 'unlearn/unlearn_accuracy': unlearn_accuracy})
                



                     


        # print("models to be updated:", model_to_update)
        # for j in range(self.data.N_slices):
        #     for index in model_to_update:

        #         self.remove_points(dataset=self.data.shards, remove_indices=indices)


        #         self.models[index].temp(dataset=self.data.slices[index][j],
        #                                 n_epochs=n_epochs[index],
        #                                 batch_size=batch_size,
        #                                 learning_rate=learning_rate,
        #                                 grad_clip=grad_clip,
        #                                 alpha=alpha,
        #                                 wandb=wandb,
        #                                 train_path=save_path)
            


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
    N_shards = 4
    N_slices = 3
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

    from utils import random_seed

    time = TimeTaking(save_path=save_path, verbose=False)

    random_seed(1)
    dataset = PTB_XL('/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/code/data/PTB_XL')

    from utils import train_test_dataset
    train_dataset, test_dataset = train_test_dataset(dataset=dataset,
                                                     test_proportion=test_proportion,
                                                     train_size=train_size,
                                                     test_size=test_size,
                                                     return_stand=False)


    test = ShardsAndSlices(train_dataset, 10, 4)

    test.contains(train_dataset.indices[0])

    test2 = Pruning(dataset=train_dataset, 
                    N_shards=N_shards, 
                    N_slices=N_slices, 
                    input_dim=input_dim, 
                    hidden_dim=hidden_dim, 
                    output_dim=output_dim, 
                    p=p, 
                    device='cpu',
                    classifier_name='logistic')

    

    acc = test2.train(n_epochs=epochs, 
                batch_size=batch_size, 
                learning_rate=learning_rate, 
                grad_clip=None, 
                alpha=0.5, 
                wandb=None, 
                save_path=save_path, 
                time_taking=time,
                test_dataset=test_dataset)
    
    print('-'*20)
    print("UNLEARNING")
    print('-'*20)
    print(acc)

    test2.unlearn(indices=train_dataset.indices[0:3],
                  n_epochs=epochs,
                  batch_size=batch_size,
                  learning_rate=learning_rate,
                  grad_clip=None,
                  alpha=0.5,
                  wandb=None,
                  save_path=save_path,
                  time_taking=time,
                  test_dataset=test_dataset
                )

    time.save()
