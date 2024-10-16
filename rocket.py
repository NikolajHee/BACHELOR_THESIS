from sklearn.linear_model import LogisticRegression
from sktime.transformations.panel.rocket import Rocket
import wandb
import os
from pathlib import Path


from base_framework.dataset import PTB_XL

data_path = os.path.join(Path.cwd(), 'PTB_XL')

def main():
    wandb.init(project="BP_final",
                name='ROCKET')
    config = wandb.config

    trf = Rocket(num_kernels=config.num_kernels, n_jobs=1, random_state=config.seed) 

    data = PTB_XL(data_path)

    from base_framework.utils import train_test_dataset
    train_dataset, test_dataset, D = train_test_dataset(dataset=data,
                                                        test_proportion=0.3,
                                                        train_size=3000,
                                                        test_size=2000,
                                                        seed=config.seed,
                                                        return_stand=False)

    D_train = train_dataset[:] #data[train_dataset.indices]
    D_test = test_dataset[:] #data[test_dataset.indices]


    # the input should be in the format of:
    #   (n_instances, n_variables, n_timepoints)


    X_train = D_train[0].permute(0,2,1).numpy()
    Y_train = D_train[1].numpy()

    X_test = D_test[0].permute(0,2,1).numpy()
    Y_test = D_test[1].numpy()

    trf.fit(X_train)

    X_train_transformed = trf.transform(X_train)
    X_test_transformed = trf.transform(X_test)

    # classification:
    model = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=config.seed)

    model.fit(X_train_transformed, Y_train)

    train_accuracy = model.score(X_train_transformed, Y_train)
    test_accuracy = model.score(X_test_transformed, Y_test)

    wandb.log({"train_accuracy": train_accuracy, 
               "test_accuracy": test_accuracy})






sweep_configuration = {
    "method": "grid",
    "metric": {"name": "test_accuracy", "goal": "maximize"},
    "parameters": {
        'num_kernels' : {"values": [2, 4, 6, 8, 10, 12, 14, 16, 32, 44, 48, 56, 64, 128, 256, 512, 1024]},
        'seed' : {"values": [0,1,2,3,4,5,6,7,8,9,10]}
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="BP_final")

wandb.agent(sweep_id, function=main)
