#!/bin/bash
#SBATCH --gres=gpu:gtx:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --output=/media/compute/homes/dmindlin/GANterfactual/slurm_logs/cyclegan_train_resnet_cycle_consistency_more_generator.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun cyclegan_train_resnet_cycle_consistency_more_generator.job.sh