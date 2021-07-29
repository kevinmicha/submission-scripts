#!/bin/bash
module purge
module load pytorch-gpu/py3/1.6.0

export BSD500_DATA_DIR=$SCRATCH/
export BSD68_DATA_DIR=$SCRATCH/
export DIV2K_DATA_DIR=$SCRATCH/
export LOGS_DIR=$SCRATCH/
export CHECKPOINTS_DIR=$SCRATCH/
