#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=$1
#SBATCH -o out_$1
#SBATCH -e err_$1
#SBATCH -p debug
#SBATCH --nodelist $2
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:$3

# Set-up the environment.
source activate medical_dialogue
# set PYTHONPATH=./

# Start the experiment.
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29505 train_gpt2.py \
--epochs 30 \
--batch_size 4 \
--log_step 100 \
--eval_all_checkpoints \
--gradient_accumulation 4  \
--num_workers 4 \
--vocab_path 'vocabulary/vocab.txt' \
--log_path 'log/'$1'.log' \
--writer_dir 'tensorboard_summary/tensorboard_'$1'/' \
--train_raw_path '../data/train_argumentation_new.txt' \
--dev_raw_path '../data/dev_argumentation_new.txt'  \
--train_tokenized_path '../data/train_$4.txt' \
--dev_tokenized_path '../data/dev_$4.txt' \
--dialogue_model_output_path 'model/'$1'_model/' \
--pretrained_model '../model/gpt2_argue_allloss_ft2_model/model_epoch4/' \
--inference_type 'groundtruth' \
--inference_result $1.json
--ft2
EOT