#!/bin/bash

#SBATCH --job-name="MAS"
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 10:00:00
#SBATCH --output=MAS_%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
nvidia-docker run -v /home/dgoyal_mt/MAS-Pytorch1/MAS-Pytorch:/home/ dgoyal_mt/mas:0.1
cd ..
cd /home
python main.py
