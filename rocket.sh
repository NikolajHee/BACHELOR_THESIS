#!/bin/sh 
### General options
### –- specify queue --
#BSUB -q gpua40
### -- set the job Name --
#BSUB -J ROCKET
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 10:00
# request 5GB of system-memory
# BSUB -R "rusage[mem=2GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o rocket_%J.out
#BSUB -e rocket_%J.err
# -- end of LSF options --


module load scipy/1.9.1-python-3.9.14

source venv/bin/activate

python3 rocket.py
