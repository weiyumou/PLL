#!/bin/bash

#SBATCH --job-name=HiBERT-wiki
#SBATCH --account=medhonda501f19w20_class
#SBATCH --partition=gpu
#SBATCH --time=14-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4gb
#SBATCH --mail-type=BEGIN,END,FAIL

# Load modules
module load python3.7-anaconda cuda/10.0.130 cudnn/10.0-v7.6
module list

# Run code
conda activate pytorch-nlp
# pip install --user tb-nightly future
# pip install --user transformers
cd /gpfs/accounts/medhonda501f19w20_class_root/medhonda501f19w20_class/yumouwei/PLL
python run_pll.py --deterministic --data_root ../wikipedia/en_wiki_out.txt  --sents_per_doc 32 --max_seq_len 64 --num_derangements 9 --max_steps 1000000 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --logging_steps 2000 --save_steps 2000

