#!/bin/bash

#SBATCH -J whole_loader
#SBATCH -c 6                           
#SBATCH -t 15:00:00
#SBATCH --mem=128g
#SBATCH -o whole.out
#SBATCH -e whole.err

# Run python script
python3 whole_train_loader.py >> whole_train.log