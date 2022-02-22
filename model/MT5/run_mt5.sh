#!/bin/bash
CUR_DATA_DIR=$DATA_DIR
port=$(($(date +%N)%30000))
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=$2
#SBATCH -o ../common/out_$2
#SBATCH -e ../common/err_$2
#SBATCH -p debug
#SBATCH --nodelist $1
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G

# Set-up the environment.
#source activate t5
# set PYTHONPATH=./
 
# Start the experiment.
#CUDA_VISIBLE_DEVICES=0,2,3 python train_mbart.py --device 0
python -m torch.distributed.launch --nproc_per_node=1 --master_port $port train_mt5.py \
--epochs 30 \
--log_step 100 \
--dialogue_model_output_path '../common/model/$2_model/' \
--eval_all_checkpoints \
--log_path '../common/log/$2.log' \
--train_path '../data/train_human_annotation.txt' \
--val_path '../data/dev_human_annotation.txt' \
--test_path '../data/test_human_annotation.txt' \
--save_path '../common/output/$2.json' \
--writer_dir '../common/tensorboard_summary/$2/' \
--generate_type 'end2end' \
--task 'nlu' \
$3
#--pretrained_model 'model/medmt5_ft1_model/model_epoch30/'
#train_argumentation_new.txt
#medmt5_ft1_model
#train_trans_argumentation.txt
#model/mt5_argue_nlu_ft2_model/model_epoch27/
#train_natural_perturbation.txt
#train_human_annotation.txt
#python evaluate_dst_mt5.py --save_path 'result/pretrained_mt5_de_en_cross.json' --ontology_path 'data/ontology/ontology_dstc2_en.json'
EOT