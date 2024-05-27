"""
Amnesiac Unlearning
"""


import os
import torch
import pickle
import glob
import warnings
from torch import nn
from tqdm import tqdm
import numpy as np
from classifier2 import classifier_train
from clustering import tsne
import torch.nn.functional as F
from torch.utils.data import DataLoader


from TS2VEC import TS2VEC, RandomCropping, hierarchical_contrastive_loss

class AmnesiacTraining(TS2VEC):
    """
    
    """
    def __init__(self, 
                 input_dim:int,
                 hidden_dim:int,
                 output_dim:int,
                 p:float,
                 device,
                 sensitive_points):
        
        super().__init__(input_dim=input_dim,
                         hidden_dim=hidden_dim,
                         output_dim=output_dim,
                         p=p,
                         device=device)

        self.sensitive_points = sensitive_points

        self.before, self.after, self.delta = {}, {}, {}

    def train(self,
              train_dataset,
              test_dataset,
              n_epochs:int, 
              batch_size:int,
              learning_rate:float,
              grad_clip,
              alpha:float,
              wandb,
              train_path:str,
              t_sne:bool,
              classifier:str):

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        


        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        if t_sne or classifier:
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # test the representations by clustering (with self-organizing maps)
        if t_sne:
            print("Clustering...")
            fig_train, fig_test = tsne(self.model,
                                       train_dataloader=train_dataloader,
                                       test_dataloader=test_dataloader,
                                       device=self.device)
            
            wandb.log({"t_sne/train_t_sne": fig_train, "t_sne/test_t_sne": fig_test})

        # test the representations by classifying the labels
        if classifier:
            print("Classifying...")
            train_accuracy, test_accuracy, baseline = classifier_train(classifier, 
                                                                       self.model, 
                                                                       train_loader=train_dataloader, 
                                                                       test_loader=test_dataloader, 
                                                                       device=self.device)
            if wandb:
                wandb.log({"classifier/train_accuracy": train_accuracy,
                        "classifier/test_accuracy": test_accuracy, 
                        "classifier/baseline": baseline})

        # main training loop
        for epoch in range(n_epochs):
            # new shuffle for each epoch
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            # ensure training mode
            self.model.module.test, self.encoder.test = False, False

            train_loss_list = []

            # evaluating each batch
            for i, (X, Y, indices) in enumerate(train_dataloader):
                # is there any sensitive points in the batch
                contain = any([i.item() in self.sensitive_points for i in indices])

                if contain: print('detected sensitive')
                
                X = X.to(self.device) # N x T x D
                
                # random crop two overlapping augments:
                crop = RandomCropping()

                signal_aug_1, signal_aug_2, crop_l, _ = crop(X) # (N x T1 x D) & (N x T2 & D)
                
                # reset gradients
                optimizer.zero_grad()

                # input augments to model
                z1 = self.model.forward(signal_aug_1.float()) # (N x T1 x Dr)
                z2 = self.model.forward(signal_aug_2.float()) # (N x T2 x Dr)
                

                # evaluate the loss on the overlapping part
                # TODO: Why only on the overlapping part?
                train_loss = hierarchical_contrastive_loss(z1[:, -crop_l:],  z2[:,:crop_l], alpha=alpha)

                if contain:
                    before = {}
                    # before weights
                    for param_tensor in self.model.state_dict():
                        if "weight" in param_tensor or "bias" in param_tensor:
                            before[param_tensor] = self.model.state_dict()[param_tensor].clone()
    
                # calculate gradients
                train_loss.backward()
                train_loss_list.append(train_loss.item())

                # clip gradient
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                
                # take gradient step
                optimizer.step()

                # Update the paraemters of the model
                self.model.update_parameters(self.encoder)
                if contain:
                    after = {}
                    # after weights
                    for param_tensor in self.model.state_dict():
                        if "weight" in param_tensor or "bias" in param_tensor:
                            after[param_tensor] = self.model.state_dict()[param_tensor].clone()

                    step = {}
                    for key in before.keys():
                        step[key] = after[key] - before[key]
                
                
                    f = open(os.path.join(train_path, f"e{epoch}b{i:04}.pkl"), "wb")
                    pickle.dump(step, f)
                    f.close()
            
            print('-'*20)
            print(f"Epoch {epoch}. Train loss {np.mean(train_loss_list)}.")
            print('-'*20)
            
            # ensure model test-mode
            self.model.module.test, self.encoder.test = True, True


            # TODO: Fix below
            # log the loss in WandB
            if wandb:
                wandb.log({"tsloss/hierarchical_contrastive_loss": np.mean(train_loss_list)})

            # test the representations by classifying the labels
            if classifier and ((epoch%10==0) or (epoch == n_epochs-1)):
                print("Classifying...")
                train_accuracy, test_accuracy, baseline = classifier_train(classifier, 
                                                                           self.model, 
                                                                           train_loader=train_dataloader, 
                                                                           test_loader=test_dataloader, 
                                                                           device=self.device)

                wandb.log({"classifier/train_accuracy": train_accuracy, 
                           "classifier/test_accuracy": test_accuracy, 
                           "classifier/baseline": baseline})
                


            # test the representations by clustering (with self-organizing maps)
            if t_sne and ((epoch%10==0) or (epoch == n_epochs-1)):
                print("Clustering...")
                fig_train, fig_test = tsne(self.model,
                                    train_dataloader=train_dataloader,
                                    test_dataloader=test_dataloader,
                                    device=self.device)
                
                wandb.log({"t_sne/train_t_sne": fig_train, "t_sne/test_t_sne": fig_test})



            # save the actual model
            torch.save(self.model.state_dict(), os.path.join(train_path, 'model.pt'))


    def unlearn(self, save_path):
        """
        Unlearns all the sensitive points
        """

        gradients = glob.glob(os.path.join(save_path, '/*.pkl'))

        sorted(gradients)

        for grad in gradients:
            path = os.path.join(save_path, grad)

            f = open(path, "rb")

            steps = pickle.load(f)

            f.close()
            
            const = 1
            
            with torch.no_grad():
                state = self.encoder.state_dict()
                for param_tensor in state:
                    if "weight" in param_tensor or "bias" in param_tensor:
                        state[param_tensor] = state[param_tensor] - const*steps[param_tensor]



    def fine_tune(self,
              train_dataset,
              test_dataset,
              optimizer,
              n_epochs:int, 
              batch_size:int,
              grad_clip,
              alpha:float,
              wandb,
              train_path:str,
              t_sne:bool,
              classifier:str):
        pass
    
if __name__ == '__main__':
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AmnesiacTraining(12, 32, 320, 0.5, DEVICE, [0])


    from dataset import PTB_XL_2
    #dataset = PTB_XL('/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/BACHELOR_THESIS/PTB_XL')

    from dataset import AEON_DATA
    dataset = AEON_DATA('ElectricDevices')

    # create train/test-split
    from utils import train_test_dataset
    train_dataset, test_dataset = train_test_dataset(dataset=dataset,
                                                     test_proportion=0.3,
                                                     train_size=10,
                                                     test_size=10,
                                                     seed=0,
                                                     return_stand=False)
    

    model.train(train_dataset=train_dataset,
                test_dataset=test_dataset,
                n_epochs=10,
                batch_size=5,
                learning_rate=0.001,
                grad_clip=None,
                wandb=None,
                alpha=0.5,
                train_path=None,
                t_sne=False,
                classifier='logistic',)
    

    model.unlearn()