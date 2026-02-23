#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gres=gpu:1 
#SBATCH --time=00-02:00:00
#SBATCH --account=ctb-lplevass
#SBATCH --job-name=Latino_Algo_1
#SBATCH --output=/home/sammys15/scratch/PhD_Project_Scratch/LATINO-PRO_Implementation/Sbatch_Outputs/Latino_Algo_1%x-%j.log
#SBATCH --error=/home/sammys15/scratch/PhD_Project_Scratch/LATINO-PRO_Implementation/Sbatch_Outputs/Latino_Algo_1%x-%j.err
#SBATCH --mail-user="sammy.sharief@umontreal.ca"
#SBATCH --mail-type=ALL

echo "Current directory: $(pwd)"
module purge
module load StdEnv/2023
module load gcc arrow/23.0.1 python/3.11
source $HOME/latino_env/bin/activate
echo "Current directory: $(pwd)"
nvidia-smi

python run_latino_16_scale_factor.py
