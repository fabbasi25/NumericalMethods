#!/bin/bash
#SBATCH --job-name= "myjob"
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=30
#SBATCH -p defq
module load python
python myjob.py
