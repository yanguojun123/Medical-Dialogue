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
source activate t5
cd GPT2-chitchat
# set PYTHONPATH=./

# Start the experiment.
#CUDA_VISIBLE_DEVICES=0,2,3 python train_mbart.py --device 0
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29518 train_mt5.py \
--epochs 30 \
--batch_size 2 \
--log_step 100 \
--dialogue_model_output_path 'model/$1_model/' \
--eval_all_checkpoints \
--log_path 'log/$1.log' \
--train_path 'data/train_$4.txt' \
--val_path 'data/dev_$4.txt' \
--test_path 'data/test_human_annotation.txt' \
--inference_result 'output/$1.json' \
--writer_dir 'tensorboard_summary/$1/' \
--gradient_accumulation 8 \
--ft2 \
--inference_type 'groundtruth' \
--task 'dpl' \
--pretrained_model 'model/mt5_argue_ppl_model/model_epoch13/'
#train_argumentation_new.txt
#medmt5_ft1_model
#train_trans_argumentation.txt
#
#python evaluate_dst_mt5.py --save_path 'result/pretrained_mt5_de_en_cross.json' --ontology_path 'data/ontology/ontology_dstc2_en.json'
EOT