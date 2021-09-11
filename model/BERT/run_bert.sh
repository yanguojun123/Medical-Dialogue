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



# Start the experiment7
allennlp train $3.json --include-package $4 -s tmp/$1
#CUDA_VISIBLE_DEVICES=3  allennlp evaluate 'tmp/bert_com_argue_pl/' '../data/0831/test_knowledge_num5.txt'  --batch-size 20 --cuda-device 0 --include-package pl --output-file 'tmp/bert_com_argue_pl/metrics.json'
EOT