# M^2-MedDialog: A Dataset and Benchmarks for Multi-domain Multi-service Medical Dialogues
# 1. Introduction 
Medical dialogue systems aim to assist doctors and patients with a range of professional medical services, i.e., diagnosis, consultation, and treatment. However, one-stop is still unexplored because: (1) no dataset has so large-scale dialogues contains both multiple medical services and fine-grained medical labels (i.e., intents, slots, values); (2) no model has addressed a based on multiple-service conversations in a unified framework.
In this work, we first build a M^2-MedDialog dataset, which contains 1,557 conversations between doctors and patients, covering 276 types of diseases, 2,468 medical entities, and 3 specialties of medical services. It has 5 intents, 7 actions, 20 slots and 2,468 candidate values in medical domain. To the best of our knowledge, it is the only medical dialogue dataset that includes both multiple medical services and fine-grained medical labels. Then, we formulate a one-stop MDS as a sequence-to-sequence generation problem. We unify a MDS with causal language modeling and conditional causal language modeling, respectively. Specifically, we employ several pretrained models (i.e., BERT-WWM, BERT-MED, GPT2, and MT5) and their variants to get benchmarks on M^2-MedDialog dataset. We also propose pseudo labeling and natural perturbation methods to expand M^2-MedDialog dataset and enhance the state-of-the-art pretrained models.
We demonstrate the results achieved by the benchmarks so far through extensive experiments on M^2-MedDialog. We release the dataset, the code, as well as the evaluation scripts to facilitate future research in this important research direction.
 

# 2. [Data](https://drive.google.com/drive/folders/1nxVEci21eU5KSejiWM4fwRlRELvkncpe?usp=sharing)
The following three kinds datasets are fine tuned in order during training.

1) train_pseudo_labeling.txt,dev_pseudo_labeling.txt: We use pseudo labeling algorithm to automatically label large-scale conversations called M^2-MedDialog-large.
2) train_natural_perturbation.txt,dev_natural_perturbation.txt: We use three methods of natural perturbation to enhance labeled date.
3) train_human_annotation.txt,dev_human_annotation.txt,test_human_annotation.txt: We process the manually labeled data to get the dataset M^2-MedDialog-base.

# 3. Requirements
Install the requirements within enviroment via pip:

`pip install -r requirements.txt`

# 4. Data proprocessing
In the directory, we put related code about processing data to get the corresponding dataset.
The following python commands are used:

1) pseudo_labeling.

We use it to automatically label our large-scale conversations to get pseudo_labeling dataset.

    `python pseudo_labeling.py`

2) natural_perturbation.

We use three strategies to build natural_pertubation dataset.

    `python natural_perturbation.py`

3) human_annotation.

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
