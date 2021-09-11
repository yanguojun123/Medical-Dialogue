#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=eva_$1
#SBATCH -o GPT2-chitchat/output/result/out_$1_new
#SBATCH -e GPT2-chitchat/output/result/err_$1_new
#SBATCH -p debug
#SBATCH --nodelist gpu06
#SBATCH --time=20-00:00:00

# Set-up the environment.
source activate meddg
cd GPT2-chitchat
# set PYTHONPATH=./

# Start the experiment7
python eva_new.py --evaluation_path output/$1 --generate_evaluation
EOT
