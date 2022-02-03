#!/bin/bash
#SBATCH -p cpu
#SBATCH -c2
#SBATCH --time=1:00:00
#SBATCH --output=/media/compute/homes/dmindlin/GANterfactual/slurm_logs/tensorboard_gan.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1

srun tensorboard_gan.job.sh