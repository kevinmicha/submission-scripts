#!/bin/bash
#SBATCH --job-name=learnlets_didn_exact_recon     # nom du job
#SBATCH --ntasks=1                   # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --gres=gpu:2                 # nombre de GPU à réserver par oeud
#SBATCH --cpus-per-task=20           # nombre de coeurs à réserver par tâche
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=20:00:00              # temps d'exécution maximum demande (HH:MM:SS)
#SBATCH --output=didn_exact_recon%j.out # nom du fichier de sortie
#SBATCH --error=didn_exact_recon%j.out  # nom du fichier d'erreur (ici commun avec la sortie)

set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. didn.sh

cd $WORK

srun --ntasks=1 python ./DIDN-master/gray_train.py --cuda --nEpochs 12 --gpus 0,1 --exact_recon

wait  # wait for all ccc_mprun(s) to complete.
