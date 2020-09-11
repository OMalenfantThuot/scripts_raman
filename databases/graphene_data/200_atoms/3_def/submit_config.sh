#!/bin/bash

#SBATCH --job-name=text_C2H6
#SBATCH --account=rrg-cotemich-ac
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=80000M
#SBATCH --time=0-10:00

../../unique_configurations.py 200at_3d 3 10 5 --output


