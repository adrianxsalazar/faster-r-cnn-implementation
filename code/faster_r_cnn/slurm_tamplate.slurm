#!/bin/bash
#SBATCH --job-name=faster_rcnn        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --output='./slurm_output/test_%A_%a.out'
#SBATCH --error='./slurm_output/test_%A_%a.err'
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)

python './code/models/faster_rcnn/faster_rcnn.py'
