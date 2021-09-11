from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import MBartForConditionalGeneration, MBartTokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from os.path import join, exists
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import shutil
import re
PAD = '[PAD]'
pad_id = 0
logger = None
import copy
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import numpy as np
import json
import re
from sklearn.metrics import f1_score, recall_score, precision_score
from typing import Iterable, Tuple, Dict, Set, List
import warnings
warnings.filterwarnings("ignore")

def longestDupSubstring(s: str) -> str:
    """
    The longest repeating string of a string
    :param s: string
    :return: str
    """
    #s = re.sub('[a-zA-z]','',s)
    nums=[ord(c)-ord('a')+1 for c in s]
    P,Q=131,2**64

    def check(L):
        visited=set()
        M=P**L%Q
        pre=0
        for i in range(L):
            pre=(pre*P+nums[i])%Q
        visited.add(pre)
        for i in range(L,n):
            pre=(pre*P+nums[i]-nums[i-L]*M)%Q
            if pre in visited:
                return i-L+1
            visited.add(pre)
        return -1

    n=len(s)
    l,r=0,n
    res=-1
    while l<r:
        mid=l+(r-l+1)//2
        index=check(mid)
        if index!=-1:
            l=mid
            res=index
        else:
            r=mid-1

    return s[res:res+l] if res!=-1 else ""

class NLTK_BLEU():
    """
    Calculate the Bleu value using the nltk toolkit
    """
    def __init__(
            self,
            smoothfunc: SmoothingFunction = None,
            ngram_weights: Iterable[float] = (0.25, 0.25, 0.25, 0.25),
            #file_path: str= ''
    ) -> None:
        """
        Class initialization
        :param smoothfunc:Selection of smoothing function during calculation
        :param ngram_weights:Indicates which blue is used
        """
        self._ngram_weights = ngram_weights
        self._scores = []
        #self.smoothfunc = SmoothingFunction().method7
        self.smoothfunc = smoothfunc
        #self.file_path = open('bleu_test/'+file_path,'w',encoding='utf-8')
        # if all(ngram_weights = SmoothingFunction().method0

    def reset(self) -> None:
        """
        Reset
        :return:NULL
        """
        self._scores = []

    # @overrides
    def get_metric(self, reset: bool = False):
        """
        get metric
        :param reset:
        :return:
        """
        score = 0.
        if len(self._scores):
            score = sum(self._scores) / len(self._scores)
        if reset:
            self.reset()
        return score

    # @overrides
    def __call__(
            self,
            references,  # list(list(str))
            hypothesis,  # list(list(str))
    ) -> None:
        for batch_num in range(len(references)):
            if len(hypothesis[batch_num]) <= 4:
                self._scores.append(0)
            else:
                self._scores.append(sentence_bleu([nltk.tokenize.word_tokenize(references[batch_num])],
                                                  nltk.tokenize.word_tokenize(hypothesis[batch_num]),
                                                  smoothing_function=self.smoothfunc, weights=self._ngram_weights))
        #self.file_path.write('\n'.join([str(x) for x in self._scores]))
        #self.file_path.close()

pattern = re.compile(r'([\u4e00-\u9fa5，]{1})\s+([\u4e00-\u9fa5，]{1})')
#args = ArgsParser().parse()
slot = ['disease', 'symptom', 'treatment', 'other', 'department', 'time', 'precaution', 'medicine', 'pathogeny',
        'side_effect',
        'effect', 'temperature', 'range_body', 'degree', 'frequency', 'dose', 'check_item', 'medicine_category',
        'medical_place', 'disease_history']
intent1 = {'Inform': 0, 'Inquire': 1, 'QuestionAnswering': 2, 'Other': 3, 'Chitchat': 4}
intent2 = {'Inform': 0, 'Inquire': 1, 'Recommend': 2, 'Diagnosis': 3, 'QuestionAnswering': 4, 'Other': 5, 'Chitchat': 6}
index = 0
intent_slot1 = {}
for x in ['Inform', 'Inquire']:
    for y in slot:
        intent_slot1['' + x + ' ' + y] = index
        index += 1
intent_slot1.update({'Inform': 40, 'Inquire': 41, 'QuestionAnswering': 42, 'Other': 43, 'Chitchat': 44})

intent_slot2 = copy.deepcopy(intent_slot1)
intent_slot2.update({'Recommend other': 45, 'Recommend medicine_category': 46, 'Recommend medical_place': 47, 'Recommend treatment': 48,
                     'Recommend precaution': 49, 'Recommend check_item': 50, 'Recommend department': 51 , 'Recommend': 52 , 'Diagnosis disease': 53,
                     'Diagnosis': 54, 'QuestionAnswering': 55})
metric = {'NLU': {},
          'AP': {},
          'RG': {}}

def exist_chinese(string):
    """
    Judge whether there is Chinese
    :param string:String to be detected
    :return:If it does not exist, return F, otherwise return Chinese characters
    """
    string = re.sub('\'\'', '', string)
    for s in string:
        if u'\u4e00' <= s <= u'\u9fff':
            return s
    return 'F'

def split_chinese(str):
    """
    Separate Chinese characters with spaces
    :param str: String to be processed (Chinese and English may be mixed)
    :return:Processed Chinese string
    """
    temp1=str.split(' ')
    res=[]
    for x in temp1:
        if exist_chinese(x):
            temp2=list(x)
            res+=temp2
        res.append(x)
    return ''.join(res)

def intent_evaluation(generatetion_list, groundtruth_list):
    """
    Evaluate the generated Intent-Slot-Value
    :param generatetion_list:Generate file list
    :param groundtruth_list:Label file list
    :return:Combination result
    """
    y_pred_intent = []
    y_gt_intent = []
    y_pred_intent_slot = []
    y_gt_intent_slot = []
    total_action = 0
    true_action = 0  # total intent number

    hyp_intent_list = []
    ref_intent_list = []
    hyp_value_list = []
    ref_value_list = []

    for index in range(len(generatetion_list)):
                y_pred_intent_temp = [0] * 5
                y_gt_intent_temp = [0] * 5
                y_pred_intent_slot_temp = [0] * 45
                y_gt_intent_slot_temp = [0] * 45
                ref_intent_temp = groundtruth_list[index]
                ref_intent_temp = re.sub('\'\'', '', ref_intent_temp)

                ref_delete_english = re.sub('[\sa-zA-Z]', '', ref_intent_temp)
                if ref_delete_english != '':
                    ref_value_list.append(split_chinese(ref_delete_english))

                ref_temp = re.sub('<\|continue\|>', '', ref_intent_temp)
                ref_intent_list.append(split_chinese(ref_temp))
                ref_intent_temp = ref_intent_temp.split('<|continue|>')

                total_action += len(ref_intent_temp)

                for i in ref_intent_temp:  # 将相应的位置变为1
                    i = re.sub('\'\'', '', i)
                    if exist_chinese(i) != 'F' and (i.split(exist_chinese(i))[0]).strip() in intent_slot1.keys():
                        y_gt_intent_slot_temp[int(intent_slot1['' + (i.split(exist_chinese(i))[0]).strip()])] = 1
                    # if exist_chinese(i)!='F':
                    if exist_chinese(i) == 'F' and i.strip() in intent_slot1.keys():
                        y_gt_intent_slot_temp[int(intent_slot1['' + i.strip()])] = 1

                    if i.strip().split(' ')[0] in intent1.keys():
                        y_gt_intent_temp[int(intent1['' + i.strip().split(' ')[0]])] = 1
                    if i.strip() in intent1.keys():
                        y_gt_intent_temp[int(intent1['' + i.strip()])] = 1
                # count+=len(ref_intent_temp)
                generate_intent = pattern.sub(r'\1\2', generatetion_list[index][1:-1])
                generate_intent = pattern.sub(r'\1\2', generate_intent)

                hyp_delete_english = re.sub('[\sa-zA-Z]', '', generate_intent)
                if ref_delete_english != '':
                    hyp_value_list.append(split_chinese(hyp_delete_english))

                hyp_temp = re.sub('<\|continue\|>', '', generate_intent)
                hyp_intent_list.append(split_chinese(hyp_temp))

                generate_intent_temp = generate_intent.split('<|continue|>')
                # generate_intent_temp = result[x]['generated_action'][y].split('<|continue|>')

                for i in generate_intent_temp:  # 将相应的位置变为1
                    if i in ref_intent_temp:
                        true_action += 1
                    i = re.sub('\'\'', '', i)
                    if exist_chinese(i) != 'F' and (i.split(exist_chinese(i))[0]).strip() in intent_slot1.keys():
                        y_pred_intent_slot_temp[int(intent_slot1['' + (i.split(exist_chinese(i))[0]).strip()])] = 1
                    if exist_chinese(i) == 'F' and i.strip() in intent_slot1.keys():
                        y_pred_intent_slot_temp[int(intent_slot1['' + i.strip()])] = 1

                    if i.strip().split(' ')[0] in intent1.keys():
                        y_pred_intent_temp[int(intent1['' + i.strip().split(' ')[0]])] = 1
                    if i.strip() in intent1.keys():
                        y_pred_intent_temp[int(intent1['' + i.strip()])] = 1
                # success_count+=len(ref_intent_temp.intersection(generate_intent_temp))

                y_pred_intent.append(y_pred_intent_temp)
                y_gt_intent.append(y_gt_intent_temp)
                y_pred_intent_slot.append(y_pred_intent_slot_temp)
                y_gt_intent_slot.append(y_gt_intent_slot_temp)

    y_pred_intent = np.array(y_pred_intent)
    y_gt_intent = np.array(y_gt_intent)
    y_gt_intent_slot = np.array(y_gt_intent_slot)
    y_pred_intent_slot = np.array(y_pred_intent_slot)

    bleu1 = NLTK_BLEU(ngram_weights=(1, 0, 0, 0), smoothfunc=SmoothingFunction().method0)
    #bleu4 = NLTK_BLEU(ngram_weights=(0, 0, 0, 1), smoothfunc=SmoothingFunction().method0)
    bleu1_value = NLTK_BLEU(ngram_weights=(1, 0, 0, 0), smoothfunc=SmoothingFunction().method0)
    bleu1(ref_intent_list, hyp_intent_list)
    #bleu4(ref_intent_list, hyp_intent_list)
    bleu1_value(ref_value_list, hyp_value_list)

    #output f1
    print("pl")
    for x in range(y_gt_intent.shape[1]):
        temp = ''
        acc = 0
        recall = 0
        for y in intent1.keys():
            if intent1[y] == x:
                temp = y
        pre_true = sum(y_pred_intent[:,x]*y_gt_intent[:,x])
        pre = sum(y_pred_intent[:,x])
        true = sum(y_gt_intent[:,x])
        acc = pre_true/pre
        rec = pre_true/true
        #print(temp, f1_score(y_gt_intent[:,x],y_pred_intent[:,x], average='macro'))
        print(temp, 2*acc*rec/(acc+rec))

    metric['NLU']['intent_micro'] = f1_score(y_gt_intent, y_pred_intent, average='micro')
    metric['NLU']['intent_macro'] = f1_score(y_gt_intent, y_pred_intent, average='macro')
    metric['NLU']['intent_weighted'] = f1_score(y_gt_intent, y_pred_intent, average='weighted')
    metric['NLU']['intent_slot_micro'] = f1_score(y_gt_intent_slot, y_pred_intent_slot, average='micro')
    metric['NLU']['intent_slot_macro'] = f1_score(y_gt_intent_slot, y_pred_intent_slot, average='macro')
    metric['NLU']['intent_slot_weighted'] = f1_score(y_gt_intent_slot, y_pred_intent_slot, average='weighted')
    metric['NLU']['bleu1'] = bleu1.get_metric(reset=False)
    metric['NLU']['accuracy'] = true_action / total_action
    #metric['AP']['bleu4'] = bleu4.get_metric(reset=False)
    metric['NLU']['combined'] = (metric['NLU']['intent_slot_micro'] * 0.5) + (0.5 * metric['NLU']['bleu1'])
    print(metric)
    #metrics_dict = compute_metrics(references=['ref_action.txt'], hypothesis='hyp_action.txt')
    # print(f1_score(y_gt_intent_slot, y_pred_intent_slot, average=None))
    return metric['NLU']['combined']

def action_evaluation(generatetion_list, groundtruth_list):
    """
    Evaluate the generated Action-Slot-Value
    :param generatetion_list:Generate file list
    :param groundtruth_list:Label file list
    :return:Combination result
    """
    y_pred_intent = []
    y_gt_intent = []
    y_pred_intent_slot = []
    y_gt_intent_slot = []
    total_action = 0
    true_action = 0  # total intent number

    hyp_intent_list = []
    ref_intent_list = []
    hyp_value_list = []
    ref_value_list = []

    for index in range(len(generatetion_list)):
                y_pred_intent_temp = [0] * 7
                y_gt_intent_temp = [0] * 7
                y_pred_intent_slot_temp = [0] * 56
                y_gt_intent_slot_temp = [0] * 56
                ref_intent_temp = groundtruth_list[index]
                ref_intent_temp = re.sub('\'\'', '', ref_intent_temp)

                ref_delete_english = re.sub('[\sa-zA-Z]', '', ref_intent_temp)
                if ref_delete_english != '':
                    ref_value_list.append(split_chinese(ref_delete_english))

                ref_temp = re.sub('<\|continue\|>', '', ref_intent_temp)
                ref_intent_list.append(split_chinese(ref_temp))
                ref_intent_temp = ref_intent_temp.split('<|continue|>')

                total_action += len(ref_intent_temp)

                for i in ref_intent_temp:  # 将相应的位置变为1
                    i = re.sub('\'\'', '', i)
                    if exist_chinese(i) != 'F' and (i.split(exist_chinese(i))[0]).strip() in intent_slot2.keys():
                        y_gt_intent_slot_temp[int(intent_slot2['' + (i.split(exist_chinese(i))[0]).strip()])] = 1
                    # if exist_chinese(i)!='F':
                    if exist_chinese(i) == 'F' and i.strip() in intent_slot2.keys():
                        y_gt_intent_slot_temp[int(intent_slot2['' + i.strip()])] = 1

                    if i.strip().split(' ')[0] in intent2.keys():
                        y_gt_intent_temp[int(intent2['' + i.strip().split(' ')[0]])] = 1
                    if i.strip() in intent2.keys():
                        y_gt_intent_temp[int(intent2['' + i.strip()])] = 1
                # count+=len(ref_intent_temp)
                generate_intent = pattern.sub(r'\1\2', generatetion_list[index][1:-1])
                generate_intent = pattern.sub(r'\1\2', generate_intent)

                hyp_delete_english = re.sub('[\sa-zA-Z]', '', generate_intent)
                if ref_delete_english != '':
                    hyp_value_list.append(split_chinese(hyp_delete_english))

                hyp_temp = re.sub('<\|continue\|>', '', generate_intent)
                hyp_intent_list.append(split_chinese(hyp_temp))

                generate_intent_temp = generate_intent.split('<|continue|>')
                # generate_intent_temp = result[x]['generated_action'][y].split('<|continue|>')

                for i in generate_intent_temp:  # 将相应的位置变为1
                    if i in ref_intent_temp:
                        true_action += 1
                    i = re.sub('\'\'', '', i)
                    if exist_chinese(i) != 'F' and (i.split(exist_chinese(i))[0]).strip() in intent_slot2.keys():
                        y_pred_intent_slot_temp[int(intent_slot2['' + (i.split(exist_chinese(i))[0]).strip()])] = 1
                    if exist_chinese(i) == 'F' and i.strip() in intent_slot2.keys():
                        y_pred_intent_slot_temp[int(intent_slot2['' + i.strip()])] = 1

                    if i.strip().split(' ')[0] in intent2.keys():
                        y_pred_intent_temp[int(intent2['' + i.strip().split(' ')[0]])] = 1
                    if i.strip() in intent2.keys():
                        y_pred_intent_temp[int(intent2['' + i.strip()])] = 1
                # success_count+=len(ref_intent_temp.intersection(generate_intent_temp))

                y_pred_intent.append(y_pred_intent_temp)
                y_gt_intent.append(y_gt_intent_temp)
                y_pred_intent_slot.append(y_pred_intent_slot_temp)
                y_gt_intent_slot.append(y_gt_intent_slot_temp)

    y_pred_intent = np.array(y_pred_intent)
    y_gt_intent = np.array(y_gt_intent)
    y_gt_intent_slot = np.array(y_gt_intent_slot)
    y_pred_intent_slot = np.array(y_pred_intent_slot)

    bleu1 = NLTK_BLEU(ngram_weights=(1, 0, 0, 0), smoothfunc=SmoothingFunction().method0)
    #bleu4 = NLTK_BLEU(ngram_weights=(0, 0, 0, 1), smoothfunc=SmoothingFunction().method0)
    bleu1_value = NLTK_BLEU(ngram_weights=(1, 0, 0, 0), smoothfunc=SmoothingFunction().method0)
    bleu1(ref_intent_list, hyp_intent_list)
    #bleu4(ref_intent_list, hyp_intent_list)
    bleu1_value(ref_value_list, hyp_value_list)

    #output f1
    print("pl")
    for x in range(y_gt_intent.shape[1]):
        temp = ''
        acc = 0
        recall = 0
        for y in intent2.keys():
            if intent2[y] == x:
                temp = y
        pre_true = sum(y_pred_intent[:,x]*y_gt_intent[:,x])
        pre = sum(y_pred_intent[:,x])
        true = sum(y_gt_intent[:,x])
        acc = pre_true/pre
        rec = pre_true/true
        #print(temp, f1_score(y_gt_intent[:,x],y_pred_intent[:,x], average='macro'))
        print(temp, 2*acc*rec/(acc+rec))

    metric['AP']['intent_micro'] = f1_score(y_gt_intent, y_pred_intent, average='micro')
    metric['AP']['intent_macro'] = f1_score(y_gt_intent, y_pred_intent, average='macro')
    metric['AP']['intent_weighted'] = f1_score(y_gt_intent, y_pred_intent, average='weighted')
    metric['AP']['intent_slot_micro'] = f1_score(y_gt_intent_slot, y_pred_intent_slot, average='micro')
    metric['AP']['intent_slot_macro'] = f1_score(y_gt_intent_slot, y_pred_intent_slot, average='macro')
    metric['AP']['intent_slot_weighted'] = f1_score(y_gt_intent_slot, y_pred_intent_slot, average='weighted')
    metric['AP']['bleu1'] = bleu1.get_metric(reset=False)
    metric['AP']['accuracy'] = true_action / total_action
    #metric['AP']['bleu4'] = bleu4.get_metric(reset=False)
    metric['AP']['combined'] = (metric['AP']['intent_slot_micro'] * 0.5) + (0.5 * metric['AP']['bleu1'])
    print(metric)
    #metrics_dict = compute_metrics(references=['ref_action.txt'], hypothesis='hyp_action.txt')
    # print(f1_score(y_gt_intent_slot, y_pred_intent_slot, average=None))
    return metric['AP']['combined']

def generate_evaluation(generation_list, groundtruth_list):
    """
    Evaluate the generated Action-Slot-Value
    :param generatetion_list:Generate file list
    :param groundtruth_list:Label file list
    :return:Combination result
    """
    hyp_list = []
    ref_list = []
    for x in range(len(generation_list)):
        hyp_list.append(' '.join(list(generation_list[x])))
        ref_list.append(' '.join(list(groundtruth_list[x])))

    #bleu1 = NLTK_BLEU(ngram_weights=(1, 0, 0, 0),smoothfunc=None)

    bleu4 = NLTK_BLEU(ngram_weights=(0, 0, 0, 1), smoothfunc=SmoothingFunction().method0)
    #bleu_aver = NLTK_BLEU(ngram_weights=(0.25, 0.25, 0.25, 0.25),smoothfunc=None)

    #bleu1(ref_list, hyp_list)
    bleu4(ref_list, hyp_list)
    return bleu4.get_metric(reset=False)
'''class MyDataset(Dataset):
    """

    """

    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = [self.data_list['inputs']['input_ids'][index],
                     self.data_list['inputs']['attention_mask'][index],
                     self.data_list['labels']['input_ids'][index],
                     self.data_list['labels']['attention_mask'][index]]
        #input_ids = [int(token_id) for token_id in input_ids.split()]
        #print(input_ids)
        return input_ids

    def __len__(self):
        return self.data_list['inputs']['input_ids'].size(0)'''

class MyDataset(Dataset):
    """
    Dataset class
    """

    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        """
        Get every item in the dataset
        :param index:Subscript indicator
        :return:Data sample
        """
        input_ids = self.data_list[index].strip()
        #input_ids = [int(token_id) for token_id in input_ids.split()]
        return input_ids

    def __len__(self):
        return len(self.data_list)

def setup_train_args():
    """
    Set training parameters
    :returns: parameter object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1', type=str, required=False, help='Set which graphics cards to use')
    parser.add_argument('--no_cuda', action='store_true', help='Training without GPU')
    parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
                        help='Select model config file')
    parser.add_argument('--vocab_path', default='vocabulary/vocab_small.txt', type=str, required=False, help='Select vocabulary file')
    parser.add_argument('--log_path', default='data/training.log', type=str, required=False, help='Training log storage location')
    parser.add_argument('--epochs', default=30, type=int, required=False, help='epoch numbers')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='learning rate')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up steps')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='How many steps to report loss')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='Gradient accumulation numbers')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--dialogue_model_output_path', default='dialogue_model/', type=str, required=False,
                        help='Dialog model output path')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='Pre training model folder to load')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard path file')
    parser.add_argument('--seed', type=int, default=5, help='random seed')
    parser.add_argument('--num_workers', type=int, default=1, help="The number of threads used by the dataloader to load data")
    parser.add_argument('--eval_all_checkpoints', action='store_true', help='Evaluate on all models')
    parser.add_argument('--train_path', default='data/train.txt', type=str, required=False, help='train dataset')
    parser.add_argument('--val_path', default='data/val.txt', type=str, required=False, help='validation dataset')
    parser.add_argument('--test_path', default='data/test.txt', type=str, required=False, help='test dataset')
    parser.add_argument('--inference_result', default='output/pretrained_mt5.txt', type=str, required=False, help='generated result file')
    parser.add_argument("--local_rank", type=int, default=-1, help='distributed')
    parser.add_argument("--ft2", action='store_true', help='second fine tune')
    parser.add_argument('--inference_type', default='groundtruth', type=str, required=True, help='generate end2end ')
    parser.add_argument('--model', default='train', type=str, required=False, help='train or test ')
    parser.add_argument('--tokenizer_path', default='tokenizer', type=str, required=False, help='tokenizer path')
    parser.add_argument('--task', default='pl', type=str, required=False, help='task: nlu,pl,nlg')
    return parser.parse_args()


def set_random_seed(args):
    """
    Set random seeds for training
    :param args: Parameter list
    :return: NULL
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_logger(args):
    """
    Output logs to log files and console
    :param args:Parameter list
    :return Logger object
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # Create a handler to write to the log file
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Create a handler to output logs to the console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

def collate_fn(batch):
    """
    Calculate the longest input of all samples in the batch, and align the length of other inputs to it
    :param batch:Batch data fetched by dataloder each time
    :return:Data in tensor format
    """
    global pad_id
    input_ids = []
    label_ids = []
    btc_size = len(batch)

    inputs = []
    labels = []
    for btc_idx in range(btc_size):
        try:#Different tasks have different inputs and outputs
            #generate intent:
            inputs.append(batch[btc_idx].split("<|intent|>")[0].split("<|endofcontext|>")[1])
            labels.append(batch[btc_idx].split("<|endofcurrentuser|>")[1].split("<|endoftext|>")[0])

            # generate action
            #inputs.append(batch[btc_idx].split("<|intent|>")[0].split("<|endoftext|>")[1])
            #labels.append(batch[btc_idx].split("<|endofcurrentuser|>")[1].split("<|endoftext|>")[0])
            #labels.append(batch[btc_idx].split("<|endofintent|>")[1].split("<|response|>")[0])

            #generate response
            #inputs.append(batch[btc_idx].split("<|response|>")[0].split("<|endoftext|>")[1])
            #inputs.append(batch[btc_idx].split('<|knowledge|>')[0].split('<|endoftext|>')[1] \
            #                       + batch[btc_idx].split('<|endofknowledge|>')[1].split('<|response|>')[0])
            #labels.append(batch[btc_idx].split("<|endofaction|>")[1].split("<|endoftext|>")[0])
        except IndexError:
                if len(inputs)> len(labels):
                    inputs.pop()
                #print(len(inputs),len(labels))
    return [inputs, labels]

def train(model, device, train_list, multi_gpu, args, tokenizer, tb_writer):
    """
    Train the model
    :param model:Pretrained model
    :param device:CPU or GPU
    :param train_list:train set
    :param multi_gpu:Is it multi GPU training
    :param args:Experimental parameters
    :param tokenizer:Tokenizer object of pre training model
    :param tb_writer:Tensorboard writer file object
    :return:NULL
    """
    train_dataset = MyDataset(train_list)
    #train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=collate_fn)
    model.train()
    # Calculate the total number of steps for parameter optimization of all epochs steps
    total_steps = int(train_dataset.__len__() * args.epochs / args.batch_size / args.gradient_accumulation)
    logger.info('total training steps = {}'.format(total_steps))

    # Set up the optimizer and use the warmup policy at the initial training
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,num_training_steps=total_steps)

    logger.info('starting training')
    # Used to count the accumulated loss of each gradient
    running_loss = 0
    # Count the total number of steps trained
    overall_step = 0
    # Number of times to record out of memory
    oom_time = 0
    # start trainning
    for epoch in range(args.epochs):
        if torch.cuda.is_available():
            sampler = DistributedSampler(train_dataset)
            sampler.set_epoch(epoch)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      collate_fn=collate_fn, sampler=sampler)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers,
                                          collate_fn=collate_fn)
        epoch_start_time = datetime.now()
        for batch_idx, input_ids in enumerate(train_dataloader):
                #print(inputs)
            #if len(input_ids[0]) and len(input_ids[1]) >0:
                inputs = tokenizer(input_ids[0], return_tensors="pt",padding=True, max_length=800, truncation=True).to(device)
                #input_ids = inputs['inputs']['input_ids'].to(device)
                #attention_mask = inputs['inputs']['attention_mask'].to(device)
                with tokenizer.as_target_tokenizer():
                    #labels_input_ids = inputs['labels']['input_ids'].to(device)
                    if input_ids[1] is not None:
                        labels = tokenizer(input_ids[1], return_tensors="pt", padding=True).to(device)
                # Solve the CUDA out of memory problem caused by insufficient video memory during operation
                try:
                    outputs = model(**inputs, labels=labels["input_ids"])
                    loss = outputs.loss
                    '''outputs = model(**inputs, decoder_input_ids=labels['input_ids'])
                    lm_logits = outputs[0]
                    shift_logits = lm_logits[..., :-1, :].contiguous()
                    shift_labels = labels['input_ids'][..., 1:].contiguous()
                    #print("logits:{}, labels:{}".format(shift_logits.size(), shift_labels.size()))
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))'''
                    #print("loss:", loss)
                    if multi_gpu:
                        loss = loss.mean()
                    if args.gradient_accumulation > 1:
                        loss = loss.mean() / args.gradient_accumulation
                    #loss.backward(loss.clone().detach())
                    loss.backward()
                    # Gradient clipping solves the problem of gradient disappearance or explosion, that is, setting the threshold
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    # After the gradient accumulation of a certain step, update the parameters
                    if (batch_idx + 1) % args.gradient_accumulation == 0:
                        running_loss += loss.mean().item()
                        # Update parameters
                        optimizer.step()
                        # Clear gradient information
                        optimizer.zero_grad()
                        # warm up
                        scheduler.step()
                        overall_step += 1
                        # Update log and tnesorboardx information
                        if (overall_step + 1) % args.log_step == 0 and args.local_rank == 0:
                            logger.info(
                                "batch {} of epoch {}, loss {}".format(
                                    batch_idx + 1, epoch + 1, loss.mean()))
                            tb_writer.add_scalar('loss', loss.mean().item(), overall_step)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        oom_time += 1
                        logger.info("WARNING: ran out of memory,times: {}".format(oom_time))
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        logger.info(str(exception))
                        raise exception
        if args.local_rank == 0:
            logger.info('saving model for epoch {}'.format(epoch + 1))
            model_path = join(args.dialogue_model_output_path, 'model_epoch{}'.format(epoch + 1))
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            os.mkdir(model_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(model_path)
            logger.info('epoch {} finished'.format(epoch + 1))
        epoch_finish_time = datetime.now()
        logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))
    logger.info('training finished')


'''def evaluate(model, device, dev_list, multi_gpu, args, tokenizer, tb_writer, overstep):
    """
    evaluate all model
    :param model:trained model
    :param device:CPU or GPU
    :param dev_list:validation set
    :param multi_gpu:Is it multi GPU training
    :param args:Experimental parameters
    :param tokenizer:Tokenizer object of pre training model
    :param tb_writer:Tensorboard writer file object
    :param overall_step:Total training steps
    :return:Total loss of the current validation set
    """
    logger.info("start evaluating model")
    model.eval()
    logger.info('starting evaluating')
    loss_all = 0
    accuracy_epoch = 0
    batch_step =0
    # tb_writer = SummaryWriter(log_dir=args.writer_dir)
    test_dataset = MyDataset(test_list)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=collate_fn)
    with torch.no_grad():

        for batch_idx, input_ids in enumerate(test_dataloader):
            #special_index = (input_ids[:, 0:6] - np.ones((input_ids.size(0), 6), dtype=int)).to(device)
            #input_ids = input_ids[:, 6:].to(device)
            # input_ids.to(device)
            if len(input_ids[0]) and len(input_ids[1]) >0:
                inputs = tokenizer(input_ids[0], return_tensors="pt", padding=True, max_length=800, truncation=True)
                inputs = inputs.to(device)

                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(input_ids[1], return_tensors="pt", padding=True)
                    labels = labels.to(device)
                    outputs = model(**inputs, labels=labels["input_ids"])
                    loss = outputs.loss


                    loss_all += loss.mean()
                    overstep[0] += 1
                    batch_step += 1
                    if multi_gpu:
                        loss = loss.mean()

                    if args.gradient_accumulation > 1:
                        loss = loss.mean() / args.gradient_accumulation

                    if (batch_idx % args.log_step) == 0:
                        logger.info(
                            "evaluate batch {} ,loss {}".format(
                                batch_idx, loss.mean()))
                    if args.local_rank == 0:
                        tb_writer.add_scalar('loss', loss.mean().item(), overstep[0])
        batch_num = len(test_dataloader)
        logger.info("finishing evaluating. loss {}".format(
            loss_all / batch_step))

    return loss_all / batch_step'''

def evaluate(model, device, test_list, args, tokenizer):
    """
    evaluate all model
    :param model:trained model
    :param device:CPU or GPU
    :param dev_list:validation set
    :param args:Experimental parameters
    :param tokenizer:Tokenizer object of pre training model
    :return:Result the current validation set
    """
    model.eval()
    logger.info('starting evaluating')
    test_dataset = MyDataset(test_list)
    generation = []
    labels = []
    nlg_labels = []
    result = 0
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=collate_fn)
    with torch.no_grad():

        for batch_idx, input_ids in enumerate(test_dataloader):
            labels += input_ids[1]
            #special_index = (input_ids[:, 0:6] - np.ones((input_ids.size(0), 6), dtype=int)).to(device)
            #input_ids = input_ids[:, 6:].to(device)
            # input_ids.to(device)
            if len(input_ids[0])>0 and len(input_ids[1]) >0:
                inputs = tokenizer(input_ids[0], return_tensors="pt", padding=True, max_length=800, truncation=True)
                inputs = inputs.to(device)

                outputs = model.generate(inputs["input_ids"], max_length=100)
                for index in range(len(outputs)):
                    if args.task == 'nlg':
                        nlg_labels.append(input_ids[1][index].split('<|response|>')[1].split('<|endofresponse|>')[0])
                    #print(tokenizer.decode(outputs[index]))
                    temp = re.sub('</s>', '', re.sub('<pad>', '', tokenizer.decode(outputs[index])))
                    if '<|action|>' in temp and '<|endofaction|>' in temp:
                        temp = temp.split('<|action|>')[1].split('<|endofaction|>')[0]
                        #print("generation:", temp)
                        generation.append(temp)
                    if '<|intent|>' in temp and '<|endofintent|>' in temp:
                        generation.append(temp)
                    if '<|response|>' in temp and '<|endofresponse|>' in temp:
                        temp = temp.split('<|response|>')[1].split('<|endofresponse|>')[0]
                        #print("generation:", temp)
                        generation.append(temp)
                    else:
                        generation.append(' ')
                #print("generation:", generation)
                #print("nlg_labels:", nlg_labels)
        if args.task == 'pl':
            result = action_evaluation(generation, labels)
        elif args.task == 'nlg':
            result = generate_evaluation(generation, nlg_labels)
        else:
            result = intent_evaluation(generation, labels)
        logger.info("finishing evaluating. result {}".format(result))
    return result

def generate(model,tokenizer,test_list,args,device):
    """
    Use model for information
    :param model:Slected best model
    :param tokenizer:Tokenizer object of pre training model
    :param test_list:Test dataset
    :param args:Experimental parameters
    :param device:CPU or GPU
    :return:NULL
    """
    logger.info('starting generating')
    save_path = open(args.inference_result, 'w', encoding='utf-8')
    #test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn1)
    joint_acc = 0
    count = 0
    model.eval()
    dialogue_all = []
    dialogue_dict = {}
    for dialogue in test_list:

        dialogue_dict[''+str(count)] = {
            'target_intent': [],
            'generated_intent': [],
            'target_action': [],
            'generated_action': [],
            'target_response': [],
            'generated_response': []
        }


        # process dialogue
        dialogue_inputs = []
        dialogue_groundtruth = []
        decoder_inputs = []
        outputs = []
        for turns in dialogue.split('\n'):
            #generate intent
            dialogue_inputs.append(turns.split("<|intent|>")[0].split("<|endofcontext|>")[1])
            dialogue_groundtruth.append(turns.split("<|endofcurrentuser|>")[1].split("<|endoftext|>")[0])

            #generate response
            #dialogue_inputs.append(turns.split('<|knowledge|>')[0].split('<|endoftext|>')[1] \
            #                      +turns.split('<|endofknowledge|>')[1].split('<|response|>')[0])
            #dialogue_inputs.append(turns.split('<|response|>')[0].split('<|endoftext|>')[1])
            #dialogue_groundtruth.append(turns.split('<|endofaction|>')[1].split('<|endoftext|>')[0])

            # generate action
            #dialogue_inputs.append(turns.split('<|intent|>')[0].split('<|endoftext|>')[1])
            decoder_inputs.append(turns.split('<|endofcurrentuser|>')[1].split('<|action|>')[0])
            #dialogue_groundtruth.append(turns.split('<|endofcurrentuser|>')[1].split('<|endoftext|>')[0])
            #dialogue_groundtruth.append(turns.split('<|endofintent|>')[1].split('<|endoftext|>')[0])

        # model generate

        inputs = tokenizer(dialogue_inputs, return_tensors="pt", padding=True).to(device)
        decoder_inputs = tokenizer(decoder_inputs, return_tensors="pt", padding=True).to(device)
        #print(inputs)
        #outputs = model.generate(inputs["input_ids"], max_length=100, forced_bos_token_id=tokenizer.encode('<en>')[0])
        if args.inference_type == 'groundtruth':
                #for index in range(len(dialogue_inputs)):
                #inputs = tokenizer(dialogue_inputs, return_tensors="pt").to(device)
                #print(decoder_inputs[index])
                #decoder_input_ids = tokenizer(decoder_inputs[index], return_tensors="pt").to(device)
                outputs= model.generate(inputs["input_ids"], decoder_input_ids=decoder_inputs,max_length=100)
                #print(outputs)
        else:
            outputs = []
            break_tokens = tokenizer.encode('</s>')

            #print('count:',count)
            ty = 'groundtruth'
            if ty == 'predicted':
                for turns in dialogue.split('\n'):
                    get_intent = False
                    get_action = False
                    inputs = tokenizer(turns.split('<|intent|>')[0].split('<|endofcontext|>')[1], return_tensors="pt").to(device)
                    knowledge = turns.split('<|endofintent|>')[1].split('<|action|>')[0]
                    #print(knowledge)

                    indexed_tokens = tokenizer.encode('<|intent|>')[:-1]
                    tokens_tensor = torch.tensor(indexed_tokens).to(device).unsqueeze(0)
                    #print(inputs, inputs['input_ids'].size(), tokens_tensor.size())
                    predicted_index = 0
                    predicted_text = ''
                    try:
                        while predicted_index != break_tokens[0]:
                            predictions = model(**inputs, decoder_input_ids=tokens_tensor)[0]
                            predicted_index = torch.argmax(predictions[0, -1, :]).item()
                            #print("pre")
                            #temp = re.sub('[^\u4e00-\u9fa5]','',tokenizer.decode(predicted_index))
                            temp = re.sub('[a-zA-Z<>|]', '', tokenizer.decode(predicted_index))
                            #print("temp:", temp)
                            if temp != '':
                                if not get_intent and temp in predicted_text:
                                    indexed_tokens += [tokenizer.encode('<|endofintent|>')[0]]
                                    #get_intent = True
                                elif get_intent and not get_action and temp in predicted_text.split('<|action|>')[1]:
                                    indexed_tokens += [tokenizer.encode('<|endofaction|>')[0]]
                                    #get_action = True
                                elif get_intent and get_action and temp in predicted_text.split('<|response|>')[1]:
                                    indexed_tokens += [tokenizer.encode('<|endofresponse|>')[0]]
                                else:
                                    indexed_tokens += [predicted_index]
                            #print(predicted_index, tokenizer.decode(predicted_index))
                            else:
                                indexed_tokens += [predicted_index]
                            #print("indexed_tokens:", indexed_tokens)
                            predicted_text = tokenizer.decode(indexed_tokens)
                            #print('predicted_text:',predicted_text)
                            '''temp = longestDupSubstring(re.sub('[^\u4e00-\u9fa5]','',predicted_text))
                            print('temp:',temp)'''
                            '''if temp != '':
                                if '<|endofintent|>' not in predicted_text:
                                    indexed_tokens += [tokenizer.encode('<|endofintent|>')[0]]
                                elif '<|endofaction|>' not in predicted_text:
                                    indexed_tokens += [tokenizer.encode('<|endofaction|>')[0]]
                                elif '<|endofresponse|>' not in predicted_text:
                                    indexed_tokens += [tokenizer.encode('<|endofresponse|>')[0]]'''
                            #print("predicted_text:", predicted_text)
                            if '<|endofintent|>' in predicted_text and not get_intent:
                                #print("predicted_text", predicted_text)
                                get_intent = True
                                # generated_intents.append('<|continue|>'.join(predicted_text.split('<|intent|>')[1].split('<|continue|>')[0:-2]))
                                indexed_tokens = tokenizer.encode(predicted_text +knowledge + '<|action|>')[:-1]

                            if '<|endofaction|>' in predicted_text and not get_action:
                                get_action = True
                                #generated_actions.append(
                                 #   '<|continue|>'.join(predicted_text.split('<|action|>')[1].split('<|continue|>')[:-2]))
                                '''indexed_tokens = tokenizer.encode(
                                    predicted_text.split('<|action|>')[0] + '<|action|> {} <|endofaction|> <|response|>'.format(
                                        '<|continue|>'.join(predicted_text.split('<|action|>')[1].split('<|continue|>')[:-2])))[1:-1]'''
                                indexed_tokens = tokenizer.encode(predicted_text + '<|response|>')[:-1]

                            predicted_text = tokenizer.decode(indexed_tokens)
                            tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
                            #print('tokens_tensor:', tokens_tensor.size())
                            if tokenizer.decode(indexed_tokens).endswith('<|endofresponse|>'):
                                break
                            if tokens_tensor.size(-1)>200:
                                indexed_tokens = tokenizer.encode(predicted_text+ ' <|endofresponse|>')
                                break
                    except RuntimeError:
                        pass
                    predicted_text = tokenizer.decode(indexed_tokens)
                    #print(predicted_text)
                    outputs.append(indexed_tokens)

            else:
                for turns in dialogue.split('\n'):
                    get_action = False
                    inputs = tokenizer(turns.split('<|intent|>')[0].split('<|endoftext|>')[1], return_tensors="pt").to(
                        device)
                    #knowledge = turns.split('<|endofintent|>')[1].split('<|action|>')[0]
                    # print(knowledge)

                    indexed_tokens = tokenizer.encode(turns.split('<|endofcurrentuser|>')[1].split('<|action|>')[0]+' <|action|>')[:-1]
                    response = turns.split('<|endofcurrentuser|>')[1].split('<|response|>')[0]
                    indexed_actions = ''
                    indexed_response = ''

                    tokens_tensor = torch.tensor(indexed_tokens).to(device).unsqueeze(0)
                    # print(inputs, inputs['input_ids'].size(), tokens_tensor.size())
                    predicted_index = 0
                    predicted_text = ''
                    try:
                        while predicted_index != break_tokens[0]:
                            predictions = model(**inputs, decoder_input_ids=tokens_tensor)[0]
                            predicted_index = torch.argmax(predictions[0, -1, :]).item()
                            #print('predicted_text:',predicted_text)
                            # print("pre")
                            # temp = re.sub('[^\u4e00-\u9fa5]','',tokenizer.decode(predicted_index))
                            temp = re.sub('[a-zA-Z<>|]', '', tokenizer.decode(predicted_index))
                            # print("temp:", temp)
                            if temp != '':
                                if not get_action and temp in predicted_text.split('<|action|>')[1]:
                                    indexed_tokens += [tokenizer.encode('<|endofaction|>')[0]]
                                    # get_action = True
                                elif  get_action and temp in predicted_text.split('<|response|>')[1]:
                                    indexed_tokens += [tokenizer.encode('<|endofresponse|>')[0]]
                                else:
                                    indexed_tokens += [predicted_index]
                            # print(predicted_index, tokenizer.decode(predicted_index))
                            else:
                                indexed_tokens += [predicted_index]
                            # print("indexed_tokens:", indexed_tokens)
                            predicted_text = tokenizer.decode(indexed_tokens)

                            if '<|endofaction|>' in predicted_text and not get_action:
                                get_action = True
                                indexed_tokens = tokenizer.encode(response + '<|response|>')[:-1]
                                indexed_actions = tokenizer.encode(predicted_text)[:-1]

                            predicted_text = tokenizer.decode(indexed_tokens)
                            tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
                            # print('tokens_tensor:', tokens_tensor.size())
                            if tokenizer.decode(indexed_tokens).endswith('<|endofresponse|>'):
                                indexed_response = tokenizer.encode(predicted_text.split('<|endofknowledge|>')[1])
                                break
                            if tokens_tensor.size(-1) > 300:
                                indexed_response = tokenizer.encode(predicted_text.split('<|endofknowledge|>')[1] + ' <|endofresponse|>')
                                break
                    except RuntimeError:
                        pass
                    #predicted_text = tokenizer.decode(indexed_tokens)
                    # print(predicted_text)
                    if len(indexed_actions)==0:
                        indexed_actions = tokenizer.encode('<|action|> <|endofaction|>')[:-1]
                    outputs.append(indexed_actions+indexed_response)
        # tokenizer decode and
        for index in range(len(outputs)):
            #print(len(outputs))
            print(tokenizer.decode(outputs[index]))
            generation = re.sub('</s>', '', re.sub('<pad>', '', tokenizer.decode(outputs[index])))
            #print("generation", generation)
            #generation = tokenizer.decode(outputs[index]).split('</s>')[0].split('<pad>')[1]
            #print("groundtruth:", dialogue_groundtruth[index])
            #dialogue_dict[''+str(count)]['target_intent'].append(dialogue_groundtruth[index].split('<|intent|>')[1].split('<|endofintent|>')[0])
            dialogue_dict[''+str(count)]['target_action'].append(dialogue_groundtruth[index].split('<|action|>')[1].split('<|endofaction|>')[0])
            dialogue_dict[''+str(count)]['target_response'].append(dialogue_groundtruth[index].split('<|response|>')[1].split('<|endofresponse|>')[0])
            if '<|intent|>' in generation and '<|endofintent|>' in generation:
                dialogue_dict[''+str(count)]['generated_intent'].append(generation.split('<|intent|>')[1].split('<|endofintent|>')[0])
            else:
                dialogue_dict[''+str(count)]['generated_intent'].append(' ')
            if '<|action|>' in generation and '<|endofaction|>' in generation:
                dialogue_dict[''+str(count)]['generated_action'].append(generation.split('<|action|>')[1].split('<|endofaction|>')[0])
            else:
                dialogue_dict[''+str(count)]['generated_action'].append(' ')
            if '<|response|>' in generation and '<|endofresponse|>' in generation:
                dialogue_dict[''+str(count)]['generated_response'].append(generation.split('<|response|>')[1].split('<|endofresponse|>')[0])
            else:
                dialogue_dict[''+str(count)]['generated_response'].append(' ')
        print("count:", count)
        count += 1
    json.dump(dialogue_dict, save_path, indent=1, ensure_ascii=False)
    save_path.close()

def init(args):
    """
    initialization
    :param args:Experimental parameters
    :return:tb_writer and multi_gpu
    """
    # Create the output directory of the dialog model
    if not os.path.exists(args.dialogue_model_output_path):
        os.mkdir(args.dialogue_model_output_path)
    # Record tensorboardx
    if os.path.exists(args.writer_dir):
        shutil.rmtree(args.writer_dir)
    os.mkdir(args.writer_dir)
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    # Whether to use multiple GPUs for parallel operation
    multi_gpu = False
    if torch.cuda.device_count() > 1:
        multi_gpu = True
    return tb_writer, multi_gpu
def main():
    """
   Main function
   :return: NULL
   """
    args = setup_train_args()

    tb_writer = ''
    multi_gpu = ''
    # Log output to both file and console
    global logger
    logger = create_logger(args)

    print("args:", args)
    # Setup CUDA, GPU & distributed training
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'

    device = torch.device("cuda", args.local_rank)

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    logger.info('using device:{}'.format(device))
    logger.info(args)


    if args.seed:
        set_random_seed(args)


    if args.pretrained_model:
        model = MT5ForConditionalGeneration.from_pretrained(args.pretrained_model)
    else:
        model = MT5ForConditionalGeneration.from_pretrained("../GPT2-chitchat/model/mt5/")
    tokenizer = MT5Tokenizer.from_pretrained("../GPT2-chitchat/model/mt5/")
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|user|>', '<|system|>','<|intent|>','<|endofintent|>',
                                                                '<|action|>','<|endofaction|>','<|response|>','<|endofresponse|>'
                                                                ,'<|knowledge|>','<|endofknowledge|>','<|continue|>','<|k|>']})

    vocab_size = len(tokenizer)
    model.to(device)
    global pad_id
    pad_id = PAD+' '
    model.to(device)

    if args.local_rank == 0:

        # Record the number of model parameters
        num_parameters = 0
        parameters = model.parameters()
        for parameter in parameters:
            num_parameters += parameter.numel()
        logger.info('number of model parameters: {}'.format(num_parameters))

        tb_writer, multi_gpu = init(args)

    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    # load model
    if args.ft2:
        logger.info("loading traing data")
        # consider train_path list
        #train_list = torch.load(args.tokenizer_path)
        train_temp = open(args.train_path, "r", encoding='utf-8').read().split('\n\n')[0:-1]
        train_list = []
        for data in train_temp:
            train_list += data.split('\n')
        logger.info("loading val data")
        val_temp = open(args.val_path, "r", encoding='utf-8').read().split('\n\n')
        val_list = []
        for data in val_temp[0:-1]:
            val_list += data.split('\n')
        logger.info("loading testing data")
        test_list = open(args.test_path, "r", encoding='utf-8').read().split('\n\n')[0:-1]
    else:
        data = open(args.train_path, 'r', encoding='utf-8')
        data_train = data.read().split('\n\n')[0:-1]
        data_list = []
        for x in data_train:
            if len(x.split(' ')) < 800 and x != '':
                data_list.append(x)
        train_list = data_list[int(0.13 * len(data_list)):]
        val_list = data_list[0:int(0.13 * len(data_list))]
        data.close()
    if args.model == 'train':
        # start trainning
        train(model, device, train_list, multi_gpu, args, tokenizer, tb_writer)
        # Model validation
        if args.local_rank == 0:
            min_model = ''
            model_loss = []
            if args.eval_all_checkpoints:
                checkpoints = [args.dialogue_model_output_path + c for c in
                               sorted(os.listdir(args.dialogue_model_output_path))[1:-1]]
                logger.info("Evaluate the following checkpoints: {}".format(checkpoints))
                overstep = [0]
                min_res = 100000
                for x in range(1, args.epochs + 1):
                    checkpoint = args.dialogue_model_output_path + 'model_epoch' + str(x)
                    model = MT5ForConditionalGeneration.from_pretrained(checkpoint)
                    logger.info("Evaluate the checkpoint: {}".format(checkpoint))
                    model.resize_token_embeddings(vocab_size)
                    model.to(device)
                    result = evaluate(model, device, val_list, multi_gpu, args, tokenizer, tb_writer, overstep)
                    model_loss.append([result, checkpoint])
                    if result < min_res:
                        min_res = result
                        min_model = checkpoint
                logger.info("the best model is " + min_model)
                print("model_loss", sorted(model_loss, key=lambda model_loss: model_loss[0], reverse=True))
            tb_writer.close()
            # min_model = 'model/medmt5_com_pl_e2e_num5_seed5_pre20_model/model_epoch5/'
            model = MT5ForConditionalGeneration.from_pretrained(min_model)
            model.to(device)
            generate(model, tokenizer, test_list, args, device)
    if args.model == 'test':
        min_model = 'model/mt5_ppl_WON_ft3_model/model_epoch19/'
        model = MT5ForConditionalGeneration.from_pretrained(min_model)
        model.to(device)
        test_result1 = 'output/mt5_argue_nlu_ft3.json'
        test_result2 = 'output/mt5_argue_dpl_nono.json'
        generate(model, tokenizer, test_list, args, device)
        #generate_new(model, tokenizer, test_list, args, device,test_result1,test_result2)
if __name__ == '__main__':
    main()
