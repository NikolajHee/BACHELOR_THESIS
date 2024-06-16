#!/bin/sh 
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J Data_Pruning
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 0:30
# request 5GB of system-memory
#BSUB -R "rusage[mem=1GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o dp_%J.out
#BSUB -e dp_%J.err
# -- end of LSF options --


module load scipy/1.9.1-python-3.9.14

source venv/bin/activate

python3 unlearn.py \
--dataset PTB_XL \
-dp \
--N_shards 5 \
--N_slices 3 \
--seed 1 \
-c logistic \
--n_epochs 25 25 25  \
-sp 25 \
--N_train 3000 \
--N_test 2000 \
-od 64 \
-hd 32


