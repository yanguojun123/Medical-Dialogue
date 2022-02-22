from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import  MT5Tokenizer
#from modeling_mt5_cl import MT5ForConditionalGeneration
import modeling_mt5_cl
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
import torch.nn as nn
import torch.nn.functional as F
import data_process
warnings.filterwarnings("ignore")


class ContrastiveLossELI5(nn.Module):
    """
    The contrastive loss from SIMCLR
    """

    def __init__(self, batch_size, temperature=0.5, verbose=True):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")

        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose: print(f"sim({i}, {j})={sim_i_j}")

            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * self.batch_size,)).to(emb_i.device).scatter_(0, torch.tensor([i]), 0.0)
            if self.verbose: print(f"1{{k!={i}}}", one_for_not_i)

            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )
            if self.verbose: print("Denominator", denominator)

            loss_ij = -torch.log(numerator / denominator)
            if self.verbose: print(f"loss({i},{j})={loss_ij}\n")

            return loss_ij.squeeze(0)

        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2 * N) * loss

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
        # if all(ngram_weights = SmoothingFunction().method0s

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
    metric['NLU']['bleu1'] = bleu1_value.get_metric(reset=False)
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
    metric['AP']['bleu1'] = bleu1_value.get_metric(reset=False)
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

    """

    def __init__(self, data_list, task, input_type):
        self.data_list = data_list
        self.task = task
        self.input_type = input_type
    def __getitem__(self, index):
        inputs = ''
        labels = ''
        if self.task == 'nlu':
            if self.input_type == 'without_context':
                inputs = (self.data_list[index].split("<|intent|>")[0].split("<|endofcontext|>")[1]).strip()
            else:
                inputs = (self.data_list[index].split("<|intent|>")[0].split("<|endoftext|>")[1]).strip()

            labels = (self.data_list[index].split("<|endofcurrentuser|>")[1].split("<|endofintent|>")[0]+ ' <|endofintent|>').strip()
        elif self.task == 'pl':
            if self.input_type == 'without_context':
                inputs = (self.data_list[index].split("<|action|>")[0].split("<|endofcontext|>")[1]).strip()
            elif self.input_type == 'without_knowledge':
                inputs = (self.data_list[index].split("<|endofintent|>")[0].split("<|endoftext|>")[1]+ ' <|endofintent|>').strip()
            else:
                inputs = (self.data_list[index].split("<|action|>")[0].split("<|endoftext|>")[1]).strip()
            labels = ('<|action|> ' + self.data_list[index].split("<|action|>")[1].split("<|response|>")[0]).strip()
            # labels.append(batch[btc_idx].split("<|endofintent|>")[1].split("<|response|>")[0])
        else:
            if self.input_type == 'without_context':
                inputs = (self.data_list[index].split("<|response|>")[0].split("<|endofcontext|>")[1]).strip()
            elif self.input_type == 'without_knowledge':
                inputs = (self.data_list[index].split("<|endofintent|>")[0].split("<|endoftext|>")[
                              1] + ' <|endofintent|> <|action|>' + self.data_list[index].split('<|action|>')[1].split('<|response|>')[0]).strip()
            else:
                inputs = (self.data_list[index].split("<|response|>")[0].split("<|endoftext|>")[1]).strip()
            # # inputs.append(batch[btc_idx].split('<|knowledge|>')[0].split('<|endoftext|>')[1] \
            # #                       + batch[btc_idx].split('<|endofknowledge|>')[1].split('<|response|>')[0])
            labels = (self.data_list[index].split("<|endofaction|>")[1].split("<|endoftext|>")[0]).strip()
        input_ids = [inputs, labels]

        #input_ids = [int(token_id) for token_id in input_ids.split()]
        return input_ids

    def __len__(self):
        return len(self.data_list)

def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--vocab_path', default='vocabulary/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--train_tokenized_path', default='data/train_tokenized.txt', type=str,
                        required=False,help='将原始训练语料tokenize之后的数据的存放位置')
    parser.add_argument('--log_path', default='data/training.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--raw', action='store_true', help='是否对原始训练语料做tokenize。若尚未对原始训练语料进行tokenize，则指定该参数')
    parser.add_argument('--epochs', default=10, type=int, required=False, help='训练的轮次')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--dialogue_model_output_path', default='dialogue_model/', type=str, required=False,
                        help='对话模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='预训练的GPT2模型的路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--seed', type=int, default=5, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int, default=1, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--mmi_model_output_path', default='mmi_model', type=str, required=False, help='MMI模型保存路径')
    parser.add_argument('--eval_all_checkpoints', action='store_true', help='在所有模型上评价')
    parser.add_argument('--train_path', default='data/train.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--val_path', default='data/val.txt', type=str, required=False, help='原始验证语料')
    parser.add_argument('--test_path', default='data/test.txt', type=str, required=False, help='原始测试语料')
    parser.add_argument('--save_path', default='output/pretrained_mt5.txt', type=str, required=False, help='保存生成结果的路径')
    parser.add_argument("--local_rank", type=int, default=-1, help='distributed')
    parser.add_argument("--ft2", action='store_true', help='second fine tune')
    parser.add_argument('--generate_type', default='end2end', type=str, required=True, help='generate end2end ')
    parser.add_argument('--model', default='train', type=str, required=False, help='train or test ')
    parser.add_argument('--tokenizer_path', default='tokenizer', type=str, required=False, help='tokenizer path')
    parser.add_argument('--task', default='pl', type=str, required=False, help='task: nlu,pl,nlg')
    parser.add_argument('--evaluate_type', default='acc', type=str, required=False, help='task: nlu,pl,nlg')
    parser.add_argument('--input_type', default='', type=str, required=False, help='ablation experiment type: WOC,WOK,all')
    parser.add_argument('--cl', action='store_true', help='Add contrastive learning')
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

# def collate_fn(batch):
#     """
#     Calculate the longest input of all samples in the batch, and align the length of other inputs to it
#     :param batch:Batch data fetched by dataloder each time
#     :return:Data in tensor format
#     """
#     global pad_id
#     input_ids = []
#     label_ids = []
#     btc_size = len(batch)
#
#     inputs = []
#     labels = []
#     for btc_idx in range(btc_size):
#         try:#Different tasks have different inputs and outputs
#             #generate intent:
#             inputs.append(batch[btc_idx].split("<|intent|>")[0].split("<|endofcontext|>")[1])
#             labels.append(batch[btc_idx].split("<|endofcurrentuser|>")[1].split("<|endoftext|>")[0])
#
#             # generate action
#             #inputs.append(batch[btc_idx].split("<|intent|>")[0].split("<|endoftext|>")[1])
#             #labels.append(batch[btc_idx].split("<|endofcurrentuser|>")[1].split("<|endoftext|>")[0])
#             #labels.append(batch[btc_idx].split("<|endofintent|>")[1].split("<|response|>")[0])
#
#             #generate response
#             #inputs.append(batch[btc_idx].split("<|response|>")[0].split("<|endoftext|>")[1])
#             #inputs.append(batch[btc_idx].split('<|knowledge|>')[0].split('<|endoftext|>')[1] \
#             #                       + batch[btc_idx].split('<|endofknowledge|>')[1].split('<|response|>')[0])
#             #labels.append(batch[btc_idx].split("<|endofaction|>")[1].split("<|endoftext|>")[0])
#         except IndexError:
#                 if len(inputs)> len(labels):
#                     inputs.pop()
#                 #print(len(inputs),len(labels))
#     return [inputs, labels]

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
    train_dataset = MyDataset(train_list, args.task, args.input_type)
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=collate_fn)
    model.train()
    # 计算所有epoch进行参数优化的总步数total_steps
    total_steps = int(train_dataset.__len__() * args.epochs / args.batch_size / args.gradient_accumulation)
    logger.info('total training steps = {}'.format(total_steps))

    # 设置优化器，并且在初始训练时，使用warmup策略
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                             num_training_steps=total_steps)

    logger.info('starting training')
    # 用于统计每次梯度累计的loss
    running_loss = 0
    # 统计一共训练了多少个step
    overall_step = 0
    # 记录 out of memory的次数
    oom_time = 0
    # 开始训练
    for epoch in range(args.epochs):
        if torch.cuda.is_available():
            sampler = DistributedSampler(train_dataset)
            sampler.set_epoch(epoch)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                          num_workers=args.num_workers,
                                          sampler=sampler)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers
                                          )
        epoch_start_time = datetime.now()
        for batch_idx, input_ids in enumerate(train_dataloader):
            #print(input_ids)
            # if len(input_ids[0]) and len(input_ids[1]) >0:
            inputs = tokenizer(input_ids[0], return_tensors="pt", padding=True, max_length=800, truncation=True).to(
                device)
            # print("input_ids", inputs['input_ids'].size())
            # input_ids = inputs['inputs']['input_ids'].to(device)
            # attention_mask = inputs['inputs']['attention_mask'].to(device)
            with tokenizer.as_target_tokenizer():
                # labels_input_ids = inputs['labels']['input_ids'].to(device)
                if input_ids[1] is not None:
                    labels = tokenizer(input_ids[1], return_tensors="pt", padding=True, truncation=True).to(device)
            # 解决在运行过程中，由于显存不足产生的cuda out of memory的问题
            try:
                outputs = model(**inputs, labels=labels["input_ids"])
                loss = outputs.loss
                # print("loss:", loss)
                if multi_gpu:
                    loss = loss.mean()
                if args.gradient_accumulation > 1:
                    loss = loss.mean() / args.gradient_accumulation
                # loss.backward(loss.clone().detach())
                loss.backward()
                # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # 进行一定step的梯度累计之后，更新参数
                if (batch_idx + 1) % args.gradient_accumulation == 0:
                    running_loss += loss.mean().item()
                    # 更新参数
                    optimizer.step()
                    # 清空梯度信息
                    optimizer.zero_grad()
                    # 进行warm up
                    scheduler.step()
                    overall_step += 1
                    # 更新日志与tnesorboardX信息
                    if (overall_step + 1) % args.log_step == 0 and args.local_rank == 0:
                        logger.info(
                            "batch {} of epoch {}, loss {}".format(
                                batch_idx + 1, epoch + 1, loss.mean()))
                        tb_writer.add_scalar('loss', running_loss, overall_step)
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


def evaluate_loss(model, device, dev_list, multi_gpu, args, tokenizer, tb_writer, overstep):
    """
    evaluate all model
    :param model:trained model
    :param device:CPU or GPU
    :param dev_list:validation set
    :param args:Experimental parameters
    :param tokenizer:Tokenizer object of pre training model
    :param tb_writer: Tensorboard writer
    :param overstep: Steps of all validation
    :return:Result the current validation set
    """
    logger.info("start evaluating model")
    model.eval()
    logger.info('starting evaluating')
    # 记录tensorboardX
    loss_all = 0
    accuracy_epoch = 0
    batch_step =0
    # tb_writer = SummaryWriter(log_dir=args.writer_dir)
    test_dataset = MyDataset(dev_list, args.task, args.input_type)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    with torch.no_grad():

        for batch_idx, input_ids in enumerate(test_dataloader):
            #special_index = (input_ids[:, 0:6] - np.ones((input_ids.size(0), 6), dtype=int)).to(device)
            #input_ids = input_ids[:, 6:].to(device)
            # input_ids.to(device)
            if len(input_ids[0]) and len(input_ids[1]) > 0:
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

    return loss_all / batch_step

def evaluate_acc(model, device, dev_list, args, tokenizer):
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
    # 记录tensorboardX
    test_dataset = MyDataset(dev_list, args.task, args.input_type)
    generation = []
    labels = []
    nlg_labels = []
    result = 0
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    with torch.no_grad():

        for batch_idx, input_ids in enumerate(test_dataloader):
            labels += input_ids[1]
            # special_index = (input_ids[:, 0:6] - np.ones((input_ids.size(0), 6), dtype=int)).to(device)
            # input_ids = input_ids[:, 6:].to(device)
            # input_ids.to(device)
            if len(input_ids[0]) > 0 and len(input_ids[1]) > 0:
                inputs = tokenizer(input_ids[0], return_tensors="pt", padding=True, max_length=800, truncation=True)
                inputs = inputs.to(device)

                outputs = model.generate(inputs["input_ids"], max_length=100)
                for index in range(len(outputs)):
                    if args.task == 'nlg':
                        nlg_labels.append(input_ids[1][index].split('<|response|>')[1].split('<|endofresponse|>')[0])
                    # print(tokenizer.decode(outputs[index]))
                    temp = re.sub('</s>', '', re.sub('<pad>', '', tokenizer.decode(outputs[index])))
                    print("temp:", temp)
                    if args.task == 'nlu':
                        if '<|intent|>' in temp and '<|endofintent|>' in temp:
                            temp = temp.split('<|intent|>')[1].split('<|endofintent|>')[0]
                            generation.append(temp)
                        else:
                            generation.append(' ')

                    elif args.task == 'pl':
                        if '<|action|>' in temp and '<|endofaction|>' in temp:
                            temp = temp.split('<|action|>')[1].split('<|endofaction|>')[0]
                            # print("generation:", temp)
                            generation.append(temp)
                        else:
                            generation.append(' ')

                    else:
                        if '<|response|>' in temp and '<|endofresponse|>' in temp:
                            temp = temp.split('<|response|>')[1].split('<|endofresponse|>')[0]
                            # print("generation:", temp)
                            generation.append(temp)
                        else:
                            generation.append(' ')
                # print("generation:", generation)
                # print("nlg_labels:", nlg_labels)

        if args.task == 'nlu':
            result = intent_evaluation(generation, labels)
        elif args.task == 'pl':
            result = action_evaluation(generation, labels)
        else:
            result = generate_evaluation(generation, nlg_labels)
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
    save_path = open(args.save_path, 'w', encoding='utf-8')
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn1)
    joint_acc = 0
    count = 0
    model.eval()
    dialogue_all = []
    dialogue_dict = {}
    for dialogue in test_list:

        dialogue_dict['' + str(count)] = {
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
            if args.task == 'nlu':
                # generate intent
                ### all
                if args.input_type == 'without_context':
                    dialogue_inputs.append(turns.split("<|intent|>")[0].split("<|endofcontext|>")[1])
                else:
                    dialogue_inputs.append(turns.split("<|intent|>")[0].split("<|endoftext|>")[1])
                dialogue_groundtruth.append(turns.split("<|endofcurrentuser|>")[1].split("<|endoftext|>")[0])

            # generate action
            if args.task == 'pl':
                if args.input_type == 'without_context':
                    dialogue_inputs.append(turns.split("<|action|>")[0].split("<|endofcontext|>")[1])
                elif args.input_type == 'without_knowledge':
                    dialogue_inputs.append(turns.split("<|endofintent|>")[0].split("<|endoftext|>")[
                                               1] + ' <|endofintent|>')
                else:
                    dialogue_inputs.append(turns.split('<|action|>')[0].split('<|endoftext|>')[1])
                # decoder_inputs.append(turns.split('<|endofcurrentuser|>')[1].split('<|action|>')[0])
                dialogue_groundtruth.append('<|action|> ' + turns.split('<|action|>')[1].split('<|response|>')[0])
                # dialogue_groundtruth.append(turns.split('<|endofintent|>')[1].split('<|endoftext|>')[0])

            if args.task == 'nlg':
                if args.input_type == 'without_context':
                    dialogue_inputs.append(turns.split("<|response|>")[0].split("<|endofcontext|>")[1])
                elif args.input_type == 'without_knowledge':
                    dialogue_inputs.append(turns.split("<|endofintent|>")[0].split("<|endoftext|>")[
                                               1] + ' <|endofintent|> <|action|>' +
                                           turns.split('<|action|>')[1].split('<|response|>')[0])
                else:
                    # generate response
                    # dialogue_inputs.append(turns.split('<|knowledge|>')[0].split('<|endoftext|>')[1] \
                    #                      +turns.split('<|endofknowledge|>')[1].split('<|response|>')[0])
                    dialogue_inputs.append(turns.split('<|response|>')[0].split('<|endoftext|>')[1])
                dialogue_groundtruth.append(turns.split('<|endofaction|>')[1].split('<|endoftext|>')[0])

        # model generate

        inputs = tokenizer(dialogue_inputs, return_tensors="pt", padding=True, max_length=100).to(device)
        # decoder_inputs = tokenizer(decoder_inputs, return_tensors="pt", padding=True, max_length=100).to(device)
        print(inputs)
        # outputs = model.generate(inputs["input_ids"], max_length=100, forced_bos_token_id=tokenizer.encode('<en>')[0])
        if args.generate_type == 'end2end':
            # for index in range(len(dialogue_inputs)):
            # inputs = tokenizer(dialogue_inputs, return_tensors="pt").to(device)
            # print(decoder_inputs[index])
            # decoder_input_ids = tokenizer(decoder_inputs[index], return_tensors="pt").to(device)
            outputs = model.generate(inputs["input_ids"], max_length=200)
            # print(outputs)
        else:
            outputs = []
            break_tokens = tokenizer.encode('</s>')

            # print('count:',count)
            ty = 'groundtruth'
            if ty == 'predicted':
                for turns in dialogue.split('\n'):
                    get_intent = False
                    get_action = False
                    inputs = tokenizer(turns.split('<|intent|>')[0].split('<|endofcontext|>')[1],
                                       return_tensors="pt").to(device)
                    knowledge = turns.split('<|endofintent|>')[1].split('<|action|>')[0]
                    # print(knowledge)

                    indexed_tokens = tokenizer.encode('<|intent|>')[:-1]
                    tokens_tensor = torch.tensor(indexed_tokens).to(device).unsqueeze(0)
                    # print(inputs, inputs['input_ids'].size(), tokens_tensor.size())
                    predicted_index = 0
                    predicted_text = ''
                    try:
                        while predicted_index != break_tokens[0]:
                            predictions = model(**inputs, decoder_input_ids=tokens_tensor)[0]
                            predicted_index = torch.argmax(predictions[0, -1, :]).item()
                            # print("pre")
                            # temp = re.sub('[^\u4e00-\u9fa5]','',tokenizer.decode(predicted_index))
                            temp = re.sub('[a-zA-Z<>|]', '', tokenizer.decode(predicted_index))
                            # print("temp:", temp)
                            if temp != '':
                                if not get_intent and temp in predicted_text:
                                    indexed_tokens += [tokenizer.encode('<|endofintent|>')[0]]
                                    # get_intent = True
                                elif get_intent and not get_action and temp in predicted_text.split('<|action|>')[1]:
                                    indexed_tokens += [tokenizer.encode('<|endofaction|>')[0]]
                                    # get_action = True
                                elif get_intent and get_action and temp in predicted_text.split('<|response|>')[1]:
                                    indexed_tokens += [tokenizer.encode('<|endofresponse|>')[0]]
                                else:
                                    indexed_tokens += [predicted_index]
                            # print(predicted_index, tokenizer.decode(predicted_index))
                            else:
                                indexed_tokens += [predicted_index]
                            # print("indexed_tokens:", indexed_tokens)
                            predicted_text = tokenizer.decode(indexed_tokens)
                            # print('predicted_text:',predicted_text)
                            '''temp = longestDupSubstring(re.sub('[^\u4e00-\u9fa5]','',predicted_text))
                            print('temp:',temp)'''
                            '''if temp != '':
                                if '<|endofintent|>' not in predicted_text:
                                    indexed_tokens += [tokenizer.encode('<|endofintent|>')[0]]
                                elif '<|endofaction|>' not in predicted_text:
                                    indexed_tokens += [tokenizer.encode('<|endofaction|>')[0]]
                                elif '<|endofresponse|>' not in predicted_text:
                                    indexed_tokens += [tokenizer.encode('<|endofresponse|>')[0]]'''
                            # print("predicted_text:", predicted_text)
                            if '<|endofintent|>' in predicted_text and not get_intent:
                                # print("predicted_text", predicted_text)
                                get_intent = True
                                # generated_intents.append('<|continue|>'.join(predicted_text.split('<|intent|>')[1].split('<|continue|>')[0:-2]))
                                indexed_tokens = tokenizer.encode(predicted_text + knowledge + '<|action|>')[:-1]

                            if '<|endofaction|>' in predicted_text and not get_action:
                                get_action = True
                                # generated_actions.append(
                                #   '<|continue|>'.join(predicted_text.split('<|action|>')[1].split('<|continue|>')[:-2]))
                                '''indexed_tokens = tokenizer.encode(
                                    predicted_text.split('<|action|>')[0] + '<|action|> {} <|endofaction|> <|response|>'.format(
                                        '<|continue|>'.join(predicted_text.split('<|action|>')[1].split('<|continue|>')[:-2])))[1:-1]'''
                                indexed_tokens = tokenizer.encode(predicted_text + '<|response|>')[:-1]

                            predicted_text = tokenizer.decode(indexed_tokens)
                            tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
                            # print('tokens_tensor:', tokens_tensor.size())
                            if tokenizer.decode(indexed_tokens).endswith('<|endofresponse|>'):
                                break
                            if tokens_tensor.size(-1) > 200:
                                indexed_tokens = tokenizer.encode(predicted_text + ' <|endofresponse|>')
                                break
                    except RuntimeError:
                        pass
                    predicted_text = tokenizer.decode(indexed_tokens)
                    # print(predicted_text)
                    outputs.append(indexed_tokens)

            else:
                for turns in dialogue.split('\n'):
                    get_action = False
                    inputs = tokenizer(turns.split('<|intent|>')[0].split('<|endoftext|>')[1], return_tensors="pt").to(
                        device)
                    # knowledge = turns.split('<|endofintent|>')[1].split('<|action|>')[0]
                    # print(knowledge)

                    indexed_tokens = tokenizer.encode(
                        turns.split('<|endofcurrentuser|>')[1].split('<|action|>')[0] + ' <|action|>')[:-1]
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
                            # print('predicted_text:',predicted_text)
                            # print("pre")
                            # temp = re.sub('[^\u4e00-\u9fa5]','',tokenizer.decode(predicted_index))
                            temp = re.sub('[a-zA-Z<>|]', '', tokenizer.decode(predicted_index))
                            # print("temp:", temp)
                            if temp != '':
                                if not get_action and temp in predicted_text.split('<|action|>')[1]:
                                    indexed_tokens += [tokenizer.encode('<|endofaction|>')[0]]
                                    # get_action = True
                                elif get_action and temp in predicted_text.split('<|response|>')[1]:
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
                                indexed_response = tokenizer.encode(
                                    predicted_text.split('<|endofknowledge|>')[1] + ' <|endofresponse|>')
                                break
                    except RuntimeError:
                        pass
                    # predicted_text = tokenizer.decode(indexed_tokens)
                    # print(predicted_text)
                    if len(indexed_actions) == 0:
                        indexed_actions = tokenizer.encode('<|action|> <|endofaction|>')[:-1]
                    outputs.append(indexed_actions + indexed_response)
        # tokenizer decode and
        for index in range(len(outputs)):
            # print(len(outputs))
            print(tokenizer.decode(outputs[index]))
            generation = re.sub('</s>', '', re.sub('<pad>', '', tokenizer.decode(outputs[index])))
            # print("generation", generation)
            # generation = tokenizer.decode(outputs[index]).split('</s>')[0].split('<pad>')[1]
            # print("groundtruth:", dialogue_groundtruth[index])
            if args.task == 'nlu':
                dialogue_dict['' + str(count)]['target_intent'].append(
                    dialogue_groundtruth[index].split('<|intent|>')[1].split('<|endofintent|>')[0])
            elif args.task == 'pl':
                dialogue_dict['' + str(count)]['target_action'].append(
                    dialogue_groundtruth[index].split('<|action|>')[1].split('<|endofaction|>')[0])
            else:
                dialogue_dict['' + str(count)]['target_response'].append(
                    dialogue_groundtruth[index].split('<|response|>')[1].split('<|endofresponse|>')[0])
            if '<|intent|>' in generation and '<|endofintent|>' in generation:
                dialogue_dict['' + str(count)]['generated_intent'].append(
                    generation.split('<|intent|>')[1].split('<|endofintent|>')[0])
            else:
                dialogue_dict['' + str(count)]['generated_intent'].append(' ')
            if '<|action|>' in generation and '<|endofaction|>' in generation:
                dialogue_dict['' + str(count)]['generated_action'].append(
                    generation.split('<|action|>')[1].split('<|endofaction|>')[0])
            else:
                dialogue_dict['' + str(count)]['generated_action'].append(' ')
            if '<|response|>' in generation and '<|endofresponse|>' in generation:
                dialogue_dict['' + str(count)]['generated_response'].append(
                    generation.split('<|response|>')[1].split('<|endofresponse|>')[0])
            else:
                dialogue_dict['' + str(count)]['generated_response'].append(' ')
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
    """
    args = setup_train_args()

    tb_writer = ''
    multi_gpu = ''
    # 日志同时输出到文件和console
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
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    # 为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    # 当得到比较好的结果时我们通常希望这个结果是可以复现
    if args.seed:
        set_random_seed(args)

    # 设置使用哪些显卡进行训练
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # tokenizer的字典大小
    if args.pretrained_model:
        if args.cl:
            model = modeling_mt5_cl.MT5ForConditionalGeneration.from_pretrained(args.pretrained_model)
        else:
            model = MT5ForConditionalGeneration.from_pretrained(args.pretrained_model)
    else:
        # google hunggiface link : google/mt5-small, if can't load , you can save your device
        model = MT5ForConditionalGeneration.from_pretrained('../../GPT2-chitchat/model/mt5/')
    tokenizer = MT5Tokenizer.from_pretrained("../../GPT2-chitchat/model/mt5/")
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|user|>', '<|system|>','<|intent|>','<|endofintent|>',
                                                                '<|action|>','<|endofaction|>','<|response|>','<|endofresponse|>'
                                                                ,'<|knowledge|>','<|endofknowledge|>','<|continue|>','<|k|>']})

    vocab_size = len(tokenizer)
    model.to(device)
    global pad_id
    pad_id = PAD+' '
    model.to(device)

    if args.local_rank == 0:

        # 记录模型参数数量
        num_parameters = 0
        parameters = model.parameters()
        for parameter in parameters:
            num_parameters += parameter.numel()
        logger.info('number of model parameters: {}'.format(num_parameters))
        print('number of model parameters:', num_parameters)

        ''' if args.raw:
            preprocess_raw_data(args.train_path, args.tokenizer_path, tokenizer, 800)'''
        # 对原始数据进行预处理,将原始语料转换成对应的token_id
        tb_writer, multi_gpu = init(args)

    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    # 加载数据
    if args.ft2:
        logger.info("loading traing data")
        # consider train_path list
        # train_list = torch.load(args.tokenizer_path)
        data_argue = json.load(open('../data/argumentation_map.json', 'r', encoding='utf-8'))
        train_temp = open(args.train_path, "r", encoding='utf-8').read().split('\n\n')[0:-1]
        train_list = []
        for data in train_temp:
            train_list += data.split('\n')

        if args.cl:
            train_list = data_process.get_data(train_list, data_argue, args)
        # print(data_new_train)
        logger.info("loading val data")
        val_temp = open(args.val_path, "r", encoding='utf-8').read().split('\n\n')
        val_list = []
        for data in val_temp[0:-1]:
            val_list += data.split('\n')

        logger.info("loading testing data")
        test1_temp = open(args.test_path, "r", encoding='utf-8').read().split('\n\n')
        test_list1 = []
        for data in test1_temp[0:-1]:
            test_list1 += data.split('\n')

        test_list2 = open(args.test_path, "r", encoding='utf-8').read().split('\n\n')[0:-1]
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
        # 开始训练
        train(model, device, train_list, multi_gpu, args, tokenizer, tb_writer)
        # 模型验证
        if args.local_rank == 0:
            best_model = ''
            model_loss = []
            if args.eval_all_checkpoints:
                checkpoints = [args.dialogue_model_output_path + c for c in
                               sorted(os.listdir(args.dialogue_model_output_path))[1:-1]]
                logger.info("Evaluate the following checkpoints: {}".format(checkpoints))
                overstep = [0]
                min_res = 100000
                max_res = 0
                for x in range(1, args.epochs + 1):
                    checkpoint = args.dialogue_model_output_path + 'model_epoch' + str(x)
                    if args.cl:
                        model = MT5ForConditionalGeneration.from_pretrained(checkpoint)
                    else:
                        model = modeling_mt5_cl.MT5ForConditionalGeneration.from_pretrained(checkpoint)
                    logger.info("Evaluate the checkpoint: {}".format(checkpoint))
                    model.resize_token_embeddings(vocab_size)
                    model.to(device)
                    if args.evaluate_type == 'loss':
                        result = evaluate_loss(model, device, val_list, multi_gpu, args, tokenizer, tb_writer, overstep)
                        model_loss.append([result, checkpoint])
                        if result < min_res:
                            min_res = result
                            best_model = checkpoint
                    else:
                        result = evaluate_acc(model, device, val_list, args, tokenizer)
                        model_loss.append([result, checkpoint])
                        if result > max_res:
                            max_res = result
                            best_model = checkpoint
                logger.info("the best model is " + best_model)
                print("model_loss", sorted(model_loss, key=lambda model_loss: model_loss[0], reverse=True))
            tb_writer.close()
            # min_model = 'model/medmt5_com_pl_e2e_num5_seed5_pre20_model/model_epoch5/'
            if args.cl:
                model = MT5ForConditionalGeneration.from_pretrained(checkpoint)
            else:
                model = modeling_mt5_cl.MT5ForConditionalGeneration.from_pretrained(checkpoint)
            model.to(device)
            generate(model, tokenizer, test_list1, args, device)
    if args.model == 'test':
        #min_model = 'model/mt5_ppl_WON_ft3_model/model_epoch19/'
        min_model = args.pretrained_model
        if args.cl:
            model = MT5ForConditionalGeneration.from_pretrained(checkpoint)
        else:
            model = modeling_mt5_cl.MT5ForConditionalGeneration.from_pretrained(checkpoint)
        model.to(device)
        test_result1 = 'output/mt5_argue_nlu_ft3.json'
        test_result2 = 'output/mt5_argue_dpl_nono.json'
        generate(model, tokenizer, test_list1, args, device)
        #generate_new(model, tokenizer, test_list, args, device,test_result1,test_result2)
if __name__ == '__main__':
    main()
