#!/bin/bash
#SBATCH --job-name=learnlets_exact_recon_shear     # nom du job
#SBATCH --ntasks=1                   # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --gres=gpu:2                 # nombre de GPU à réserver par oeud
#SBATCH --cpus-per-task=20           # nombre de coeurs à réserver par tâche
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=09:00:00              # temps d'exécution maximum demande (HH:MM:SS)
#SBATCH --output=shearlets%j.out # nom du fichier de sortie
#SBATCH --error=shearlets%j.out  # nom du fichier d'erreur (ici commun avec la sortie)

set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. learnlets.sh

cd $WORK

srun --ntasks=1 python ./learning_wavelets/training_scripts/learnlet_training.py -e -gpus 01

wait  # wait for all ccc_mprun(s) to complete.
