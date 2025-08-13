#!/bin/bash
#SBATCH --job-name=lorentz63-1000
#SBATCH --output=logs/job%j.log
#SBATCH --error=logs/job%j.err
#SBATCH --time=22:00:00
#SBATCH --partition=P100
#SBATCH --gpus=1

python main.py --problem lorentz63