#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=$1
#SBATCH -o out_$1
#SBATCH -e err_$1
#SBATCH -p debug
#SBATCH --nodelist $2
#SBATCH --time=20-00:00:00
#SBATCH --gres=gpu:2
# Set-up the environment.
source activate medical_dialogue



allennlp train $3.json --include-package $4 -s $1
EOT