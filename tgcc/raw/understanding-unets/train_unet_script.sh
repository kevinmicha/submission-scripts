#!/bin/bash
#MSUB -r train_unets                # Request name
#MSUB -n 2                         # Number of tasks to use
#MSUB -c 2                         # I want 2 cores per task since io might be costly
#MSUB -x
#MSUB -T 21600                      # Elapsed time limit in seconds
#MSUB -o unet_train_%I.o              # Standard output. %I is the job id
#MSUB -e unet_train_%I.e              # Error output. %I is the job id
#MSUB -q v100               # Queue
#MSUB -Q normal
#MSUB -m work
#MSUB -@ kmichalewicz@fi.uba.ar:begin,end
#MSUB -A 101197                  # Project ID
​
set -x
cd $WORK
​
. ./submission_scripts/env_config.sh
​
ccc_mprun -E '--exclusive' -n 1 python3 ./learning_wavelets/training_scripts/unet_training.py -gpus 01 
​
wait  # wait for all ccc_mprun(s) to complete.
