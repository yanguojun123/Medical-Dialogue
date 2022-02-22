#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=eva_$1
#SBATCH -o ../output/result/out_$1_new
#SBATCH -e ../output/result/err_$1_new
#SBATCH -p debug
#SBATCH --nodelist gpu06
#SBATCH --time=20-00:00:00

# Set-up the environment.
# set PYTHONPATH=./

# Start the experiment7
python eva.py --evaluation_path ../output/$1 --evaluation_task=nlu
EOT
