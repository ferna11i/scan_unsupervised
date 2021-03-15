#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:45:00
#SBATCH --job-name moco
#SBATCH --output=moco.txt
#SBATCH -A soscip-3-040
#SBATCH --mail-type=FAIL

module load anaconda3
source activate clusterEnv

python ../moco_upd.py --config_env ../configs/env.yml --config_exp ../configs/pretext/moco_tabledb.yml

source deactivate