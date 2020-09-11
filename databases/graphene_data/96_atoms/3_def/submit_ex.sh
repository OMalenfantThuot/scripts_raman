#!/bin/bash

#SBATCH --job-name=text_C2H6
#SBATCH --account=rrg-cotemich-ac
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2048M
#SBATCH --time=0-00:10

python example.py

