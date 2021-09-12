# M^2-MedDialog: A Dataset and Benchmarks for Multi-domain Multi-service Medical Dialogues
# 1. Introduction 
In this work, we create a multiple-domain multiple-service dataset with fine-grained medical labels for one-stop MDS.
We fit NLU, DPL and NLG into a unified SeqMDS framework, based on which, we deploy several cutting-edge pretrained language models as benchmarks.
Besides, we have introduced two data argumentation methods, i.e., pseudo labeling and natural perturbation, to generate synthetic data to enhance the model performance.
Extensive experiments have demonstrated that SeqMDS can achieve good performance with different pretrained models as backends.

# 2. Data [[link]](https://drive.google.com/drive/folders/1nxVEci21eU5KSejiWM4fwRlRELvkncpe?usp=sharing)
The following three kinds datasets are fine tuned in order during training.

1) [train_pseudo_labeling.txt,dev_pseudo_labeling.txt](http://xxx): We use pseudo label method to automatically label large-scale conversations called M^2-MedDialog-large.
2) train_natural_perturbation.txt,dev_natural_perturbation.txt: We use three methods of natural perturbation to enhance labeled date.
3) train_human_annotation.txt,dev_human_annotation.txt,test_human_annotation.txt: Processed manually labeled dataset, is the dataset in our paper called M^2-MedDialog-base.

# 3. Requirements
Install the requirements within enviroment via pip:

`pip install -r requirements.txt`

# 4. Data proprocessing
In the directory, we put related code about processing data to get the corresponding dataset.
The following python commands are used:

1) pseudo_labeling
We use it to automatically label our large-scale conversations to get pseudo_labeling dataset.

    `python pseudo_labeling.py`

2) natural_perturbation
We use three strategies to build natural_pertubation dataset.

    `python natural_perturbation.py`

3) human_annotation
We use it to process the original annotation file into the required format.

    `python human_annotation.py`

# 5. Training & Inference & Evaluation

**BERT-WWM** and **BERT-MED**

The following slurm commands are used:

The format is 'sbatch run_bert.sh [model_result_output_file] [nodelist] [model_name] [task_name]' 

For example:

`sbatch run_bert.sh  bert-wwm_nlu gpu06 bert-wwm nlu`

**GPT2:**

The format is the 'sbatch run_gpt2.sh [model_result_output_file] [nodelist] [node_number][dataset_name] [inference_type]' and 'sbatch eva.sh [result_output_file]'.
You can use two commands to training & inference and evaluation respectively.

For example:

`sbatch run_gpt2.sh gpt2_test gpu06 4 human_annoation groundtruth`

`sbatch eva.sh gpt2_test.json`

**MT5:**

The format is the 'sbatch run_mt5.sh [model_result_output_file] [nodelist] [dataset_name] [task_name] [inference_type]' and 'sbatch eva.sh [result_output_file]'.
You can use two commands to training & inference and evaluation respectively.

For example:

`sbatch run_mt5.sh mt5_test gpu06 4 human_annotation nlu groundtruth`

`sbatch eva.sh mt5_test.json`

# 6. Visulization
You can get some pictures of relevant data results via python.

`python plot.py`

# 7. Citation
TBA
