# ReMeDi: Resources for Multi-domain, Multi-service, Medical Dialogues
# 1. Introduction 
In this work, we contribute the following resources:
- (1) A dataset contains 96,965 conversations between doctors and patients, including 1,557 conversations with fine-gained labels. It covers 843 types of diseases, 5,228 medical entities, and 3 specialties of medical services across 40 domains. To the best of our knowledge, the ReMeDi dataset is the only medical dialogue dataset that covers multiple domains and services, and has fine-grained medical labels.
- (2) Benchmark methods: (a) pretrained models (i.e., BERT-WWM, BERT-MED, GPT2, and MT5) trained, validated, and tested on the ReMeDi dataset, and (b) a self-supervised contrastive learning(SCL) method to expand the ReMeDi dataset and enhance the training of the state-of-the-art pretrained models. The paper link is [https://arxiv.org/abs/2109.00430](https://arxiv.org/abs/2109.00430)
 

# 2. Data
The dataset contains:1) [ReMeDi-large.json](https://drive.google.com/drive/folders/1nxVEci21eU5KSejiWM4fwRlRELvkncpe?usp=sharing) (The dataset is large, so we save it with additional link.) 2) ReMeDi-base.json
The ReMeDi-base/large dataset is provided as json format:
* dialogue_id
* information (turn list)
   * turn_id
   * role (patient or doctor)
   * sentence (text information)
   * semantical_labels (semantical lables list)
      * text (text)
      * range (text start index and end index)
      * intent/action (intent or action)
      * slot 
      * value1
      * value2

And the following three kinds data are processed to train model or evaluate.

1) train_pseudo_labeling.txt,dev_pseudo_labeling.txt: We use pseudo labeling algorithm to automatically label large-scale conversations called M^2-MedDialog-large.
2) train_natural_perturbation.txt,dev_natural_perturbation.txt: We use three methods of natural perturbation to enhance labeled date.
3) train_human_annotation.txt,dev_human_annotation.txt,test_human_annotation.txt: We process the manually labeled data to get the dataset M^2-MedDialog-base.
# 3. Requirements
Install the requirements within enviroment via pip:

`pip install -r requirements.txt`

# 4. Data processing
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

`sbatch run_gpt2.sh gpu06 gpt2_test `

`sbatch eva.sh gpt2_test.json`

**MT5:**

The format is the 'sbatch run_mt5.sh [model_result_output_file] [nodelist] [dataset_name] [task_name] [inference_type]' and 'sbatch eva.sh [result_output_file]'.
You can use two commands to training & inference and evaluation respectively.

For example:

`sbatch run_mt5.sh gpu06 mt5_test`

`sbatch eva.sh mt5_test.json`

# 6. Annotation guideline
The folder contains contains the code and guidelines for the labeling system.

# 7. License

All resources are licensed under the MIT license.

# 8. Citation
```
@inproceedings{yan2022remedi,
  title={ReMeDi: Resources for Multi-domain, Multi-service, Medical Dialogues},
  booktitle = {SIGIR},
  author={Yan, Guojun and Pei, Jiahuan and Ren, Pengjie and Ren, Zhaochun and Xin, Xin and Liang, Huasheng and de Rijke, Maarten and Chen, Zhumin},
  year={2022}
  url={https://arxiv.org/abs/2109.00430},
}
```
