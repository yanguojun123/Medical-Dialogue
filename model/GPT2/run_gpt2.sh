#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=$2
#SBATCH -o ../common/out_$2
#SBATCH -e ../common/err_$2
#SBATCH -p debug
#SBATCH --nodelist $1
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:1

# Set-up the environment.
#source activate
#conda activate t5
# set PYTHONPATH=./
# Start the experiment.
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29505 train_gpt2.py \
--epochs 30 \
--batch_size 4 \
--log_step 100 \
--eval_all_checkpoints \
--gradient_accumulation 4  \
--num_workers 4 \
--vocab_path '../common/vocabulary/vocab.txt' \
--log_path '../common/log/'$1'.log' \
--writer_dir '../common/tensorboard_summary/tensorboard_'$2'/' \
--train_raw_path '../data/train_human_annotation.txt' \
--dev_raw_path '../data/dev_human_annotation.txt' \
--train_tokenized_path '../common/tokenizer/train_test.txt' \
--dev_tokenized_path '../common/tokenizer/dev.txt' \
--dialogue_model_output_path '../common/model/'$2'_model/' \
--ft2 \
$3
#ft1_mapping_xavier_n800_model_continue_model
#WOK_n800_tokenizer_dev.txt
#gpt2_argue_allloss_ft2_model/model_epoch4
EOT