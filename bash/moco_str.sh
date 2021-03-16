#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:45:00
#SBATCH --job-name moco
#SBATCH --output=moco_str.txt
#SBATCH -A soscip-3-040
#SBATCH --mail-type=FAIL

module load anaconda3
source activate clusterEnv

python ../simclr.py --config_env ../configs/env.yml --config_exp ../configs/pretext/moco_tablestrdb.yml

source deactivate