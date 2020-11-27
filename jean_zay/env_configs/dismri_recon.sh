#!/bin/bash
module purge
module load python/3.7.5 cuda/10.1.2 cudnn/7.6.5.32-cuda-10.1 nccl/2.5.6-2-cuda
conda activate dis-mri-recon

export FASTMRI_DATA_DIR=$SCRATCH/
export OASIS_DATA_DIR=$SCRATCH/OASIS_data
export MODELS_DIR=$SCRATCH/distributed_models
