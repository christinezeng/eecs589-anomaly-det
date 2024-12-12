#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:


#SBATCH --job-name=UNSW-NB15
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=cczeng@umich.edu
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32g
#SBATCH --time=1:00:00
#SBATCH --account=cse589s011f24_class
#SBATCH --partition=spgpu
#SBATCH --output=./UNSW-NB15.log


# The application(s) to execute along with its input arguments and options:

module load python3.11-anaconda/2024.02
module load gcc/14.1.0/
python UNSW-NB15.py
