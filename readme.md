## M^2-MedDialog: A Dataset and Benchmarks for Multi-domain Multi-service Medical Dialogues
### Paper Summary 
In this work, we create a multiple-domain multiple-service dataset with fine-grained medical labels for one-stop MDS.
We fit NLU, DPL and NLG into a unified SeqMDS framework, based on which, we deploy several cutting-edge pretrained language models as benchmarks.
Besides, we have introduced two data argumentation methods, i.e., pseudo labeling and natural perturbation, to generate synthetic data to enhance the model performance.
Extensive experiments have demonstrated that SeqMDS can achieve good performance with different pretrained models as backends.

### Running experiments
#### Requirements
install the requirements within enviroment via pip:

`pip install -r requirements.txt`

#### Datasets

dialogue datasets:
1) pseudo_labeling: first fine-tuned dataset(Bsecause the size is too large, we didn't put it here.)
2) natural_perturbation: second fine-tuned dataset
3) human_annotation: third fine-tuned dataset

knowledge:
1) knowledge_entities:original complete knowledge base file
2) knowledge.json: knowledge to be used in the dataset

dialogue 
#### Data Process
In our directory /data_process, we put all our related code about process data to get our final dataset.


##### plot
We use it to draw a picture of our results.

##### Human_annotation
Process the original annotation file into the required format and add the corresponding knowledge.

`python human_annotation.py`

##### pseudo_labeling
We use it to automatically label our large-scale conversations.

##### natural_perturbation
Three strategies are used to build natural_pertubation dataset.

run the command:

`python natural_perturbation.py`
#### train&validation&inference

BERT-WWM and BERT-MED(Running the following code can complete all processes of training, validation, inference and evaluation ):

using sbatch command:
The format is 'sbatch run_bert.sh [model_result_output_file] [nodelist] [model_name] [task_name]' for example:
`sbatch run_bert.sh  bert-wwm_nlu gpu06 bert-wwm nlu`

GPT2:

using sbatch command:

The format is the 'sbatch run_gpt2.sh [model_result_output_file] [nodelist] [node_number][dataset_name] [inference_type]'

for example:
`sbatch run_gpt2.sh gpt2_test gpu06 4 human_annoation groundtruth`

MT5:

using sbatch command:

The format is the 'sbatch run_mt5.sh [model_result_output_file] [nodelist] [dataset_name] [task_name] [inference_type]'

for example:
`sbatch run_mt5.sh mt5_test gpu06 4 human_annotation nlu groundtruth`
`

#### evaluate
GPT2 and MT5:

using sbatch command:

The format is the 'sbatch eva.sh [result_output_file]'

for example:

`sbatch eva.sh gpt2_test.json`