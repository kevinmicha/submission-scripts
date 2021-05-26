#!/bin/bash
module purge
module load feature/openmpi/net/ib/openib
module load cuda/10.1.105
module load python3/3.7.5
module load gnu/8.3.0
module load mkl/19.0.5.281
​
pip install --target=$CCCWORKDIR/installed-packages/ --upgrade ./
​
export BSD500_DATA_DIR=$CCCWORKDIR/
export BSD68_DATA_DIR=$CCCWORKDIR/
export DIV2K_DATA_DIR=$CCCWORKDIR/
export LOGS_DIR=$CCCWORKDIR/
export CHECKPOINTS_DIR=$CCCWORKDIR/
