from sklearn.linear_model import LogisticRegression
from sktime.transformations.panel.rocket import Rocket
from dataset import PTB_XL
import wandb






path_to_data = '/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/code/data/PTB_XL'




def main():
    wandb.init(project="BACHELOR_THESIS",
            name='ROCKET')
    config = wandb.config

    trf = Rocket(num_kernels=config.num_kernels, n_jobs=-1, random_state=1) 

    data = PTB_XL(path_to_data)

    from utils import train_test_dataset
    train_dataset, test_dataset = train_test_dataset(dataset=data,
                                                    test_proportion=0.3,
                                                    verbose=False,
                                                    seed=0,
                                                    return_stand=False)

    D_train = train_dataset[:] #data[train_dataset.indices]
    D_test = test_dataset[:] #data[test_dataset.indices]



    # the input should be in the format of:
    #   (n_instances, n_variables, n_timepoints)


    X_train = D_train[0].transpose(0,2,1)
    Y_train = D_train[1]

    X_test = D_test[0].transpose(0,2,1)
    Y_test = D_test[1]

    trf.fit(X_train)

    X_train_transformed = trf.transform(X_train)
    X_test_transformed = trf.transform(X_test)

    # classification:
    model = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=1)

    model.fit(X_train_transformed, Y_train)

    train_accuracy = model.score(X_train_transformed, Y_train)
    test_accuracy = model.score(X_test_transformed, Y_test)

    wandb.log({"train_accuracy": train_accuracy, 
               "test_accuracy": test_accuracy})






sweep_configuration = {
    "method": "grid",
    "metric": {"name": "test_accuracy", "goal": "minimize"},
    "parameters": {
        'num_kernels' : {"values": [8, 16, 32, 64, 128, 256, 512, 1024]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="BACHELOR_THESIS")

wandb.agent(sweep_id, function=main)
