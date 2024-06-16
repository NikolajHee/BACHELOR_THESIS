# Machine Unlearning and Representation Learning
Code producing the results presented for the bachelor degree of Artificial Intelligence and Data. 



## Installation

The scripts is mainly build on pytorch and scikit-learn. ROCKET requires sktime. Some of the data requires aeon.

## Usage

The scripts were run using the files in `/scripts`. The data is assumed to lay in folder `PTB_XL/`.

## License

Information about the project's license and any relevant terms or conditions.

## Files

A list of the important files in the repository and their purposes.
- `main.py`: file for running ts2vec.
- `unlearn.py`: file for running unlearning methods.
- `time_analysis*.py`: scripts for running experiments in regards to the time analysis.
- `accuracy_analysis*.py`: scripts for running experiments in regards to the accuracy analysis and MIA.
- `plots*.py`: scripts and notebooks for getting the plots of the thesis.
- `base_framework/`: folder containing the main files of the scripts.
- `scripts/`: folder containing bash files for running hpc-jobs.
- `results/`: folder containing some of the results.
- `rocket.py`: script for training ROCKET.
- `ts2vec_accuracy`: script for running ts2vec multiple times


## How to Run

Training a ts2vec encoder on Crop with a logistic regression:

`python3 main.py --dataset Crop --classifier logistic --n_epochs 25 --normalize off --learning_rate 0.001 -bs 8 -hd 32 -od 64 --t-sne --seed 0 `


Training the data-pruning setup

`python3 ../unlearn.py --dataset PTB_XL -dp --N_shards 5 --N_slices 3 --seed 1 -c logistic --n_epochs 25 25 25  -sp 25 --N_train 3000 --N_test 2000 -od 64 -hd 32`