#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/4/15 17:17
# @Author : kzl
# @Site :
# @File : basline.py
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.models.archival import archive_model, load_archive
import os
from typing import List, Dict, Tuple, Iterable
import tempfile
import torch
from allennlp.data.data_loaders import SimpleDataLoader
from overrides import overrides
import numpy as np
from allennlp.common.params import Params
from allennlp.commands.train import train_model
from allennlp.data import Instance
from allennlp.data.fields import TextField, MultiLabelField, ListField, Field, MetadataField ,LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import seq2vec_encoder
from allennlp.nn.util import get_text_field_mask, masked_softmax
from allennlp.common.util import JsonDict
from allennlp.training.metrics import F1Measure, Average, Metric, CategoricalAccuracy
from allennlp.predictors import Predictor
#from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
#from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer, CharacterTokenizer
import torch.nn.functional as F
from DataRead import *
from allennlp.nn import util
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import re
import pickle

###
intent1 = {'Inform': 0, 'Inquire': 1, 'QuestionAnswering': 2, 'Other': 3, 'Chitchat': 4}
slot = ['disease', 'symptom', 'treatment', 'other', 'department', 'time', 'precaution', 'medicine', 'pathogeny',
        'side_effect',
        'effect', 'temperature', 'range_body', 'degree', 'frequency', 'dose', 'check_item', 'medicine_category',
        'medical_place', 'disease_history']
intent_slot1 = {}
index = 0
for x in ['Inform', 'Inquire']:
    for y in slot:
        intent_slot1['' + x + ' ' + y] = index
        index += 1
intent_slot1.update({'Inform': 40, 'Inquire': 41, 'QuestionAnswering': 42, 'Other': 43, 'Chitchat': 44})
'''
all seq2vec baseline
'''
def convert(tensor, vocab):
    result = torch.zeros(tensor.size(0), 5).to('cuda')
    for count, x in enumerate(tensor):
        for id, y in enumerate(x):
            if y == 1.:
                label = vocab.get_token_from_index(id, namespace='labels')
                if label != '' and label.split(' ')[0] in intent1.keys():
                        result[count][int(intent1[label.split(' ')[0]])] = 1.
    return result

def indices(tensor): # get same label
    #result = torch.zeros(tensor.size(0), 45).to('cuda')
    t1 = tensor[:, 0:22]
    t2 = tensor[:, 23:26]
    t3 = tensor[:, 27].unsqueeze(1)
    t4 = tensor[:, 29:35]
    t5 = tensor[:, 36:38]
    t6 = tensor[:, 40:43]
    t7 = tensor[:, 46].unsqueeze(1)
    t8 = tensor[:, 51:54]
    t9 = tensor[:, 55:57]
    t10 = tensor[:, 60].unsqueeze(1)
    t11 = tensor[:, 66].unsqueeze(1)
    return torch.cat([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11], 1)
import json
import copy
topic_num = 64
@DatasetReader.register("mds_reader")
class TextClassificationTxtReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 model: str = None,
                 max_tokens : int = None) -> None:

        super().__init__()
        self.tokenizer = tokenizer or CharacterTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.model = model
        self.max_tokens = max_tokens

    @overrides
    def _read(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
            # split train data and val data
            #start_index = 0 if 'val' in file_path else int(0.13 * len(lines))
            #end_index = int(0.13 * len(lines)) if 'val' in file_path else -1
            start_index = 0
            end_index = -1
            # add Separator
            Separator = 'intent'
            #Separator = 'intent' if "intent" in file_path else 'action'
            for line in lines[start_index:end_index]:
                if line != '':
                    #text = '<|endoftext|> <|context|> <|endofcontext|> '+line.strip().split('<|'+Separator+'|>')[0].split('<|endofcontext|>')[1]
                    text = line.strip().split('<|' + Separator + '|>')[0]
                    intent = re.sub('[\u4e00-\u9fa5]', '', line.strip().split('<|endof'+Separator+'|>')[0].split('<|'+Separator+'|>')[1])
                    intent_slot = [x.strip() for x in intent.split('<|continue|>') if x.strip() in intent_slot1.keys()]
                    tokens = self.tokenizer.tokenize(text)
                    if self.max_tokens:
                        tokens = tokens[-1*self.max_tokens:]
                    if len(intent_slot) != 0 :
                        text_field = TextField(tokens, self.token_indexers)
                        #label_intent_field = MultiLabelField(intent_list)
                        label_intent_slot_field = MultiLabelField(intent_slot)
                        #print("label_intent_field:", label_intent_field)
                        #print("label_intent_slot_field:", label_intent_slot_field)
                        fields = {'text': text_field, 'label': label_intent_slot_field}
                        yield Instance(fields)

@Model.register("simple_classifier")
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 dropout: float = None):
        super().__init__(vocab)
        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self.embedder = embedder
        self.encoder = encoder
        vocab = vocab.from_files('tmp/lstm_nlu_P2/vocabulary/')
        '''vocab.add_tokens_to_namespace(['<|endoftext|>', '<|user|>', '<|intent|>', '<|endofintent|>',
                                            '<|action|>', '<|endofaction|>', '<|response|>', '<|endofresponse|>',
                                            'Inform', 'Inquire', 'Recommend', 'Diagnosis', 'Chitchat', 'Other',
                                            'disease',
                                            'symptom', 'treatment', 'other', 'department', 'time', 'precaution',
                                            'QuestionAnswering',
                                            'medicine', 'pathogeny', 'side_effect', 'effect', 'temperature',
                                            'range_body', 'degree',
                                            'frequency', 'dose', 'check_item', 'medicine_category', 'medical_place',
                                            'disease_history',
                                            '<|context|>', '<|endofcontext|>', '<|system|>', '<|currentuser|>',
                                            '<|continue|>', '<|endofcurrentuser|>'], 'tokens')'''
        num_labels_intent = vocab.get_vocab_size("labels")
        for x in range(num_labels_intent):
            print(str(x)+' ', vocab.get_token_from_index(x, namespace='labels'))
        print(num_labels_intent)
        #print('26954', vocab.get_token_from_index(26954, namespace='tokens'))
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels_intent)
        self.vocab = vocab
        ''' self.total_pre = Average()
        self.total_true = Average()
        self.total_pre_true = Average()
        self.accuracy = Average()

        self.total_pre_macro = Average()
        self.total_true_macro = Average()
        self.total_pre_true_macro = Average()'''
        self.all_pre_intent = Average_tensor(5)
        self.all_true_intent = Average_tensor(5)
        self.all_pre_intent_slot = Average_tensor(45)
        self.all_true_intent_slot = Average_tensor(45)

        '''self.total_pre_macro = torch.zeros(num_labels)
        self.total_true_macro = torch.zeros(num_labels)
        self.total_pre_true_macro = torch.zeros(num_labels)
        self.all_pre = torch.zeros(num_labels)
        self.all_true = torch.zeros(num_labels)'''
        #self.macro_f = MacroF(num_labels)

    def forward(self,text: Dict[str, torch.Tensor],
                    label: torch.Tensor) -> Dict[str, torch.Tensor]:
            # Shape: (batch_size, num_tokens, embedding_dim)
            #print("text:",text)
            embedded_text = self.embedder(text)
            # Shape: (batch_size, num_tokens)
            mask = util.get_text_field_mask(text)
            # Shape: (batch_size, encoding_dim)
            encoded_text = self.encoder(embedded_text, mask)
            # Shape: (batch_size, num_labels)
            logits = self.classifier(encoded_text)
            # Shape: (batch_size, num_labels)
            probs = torch.sigmoid(logits)
            probs = indices(probs)


            #label = label[:, 0:probs.size(-1)]
            label = indices(label)
            #print("probs:", probs.size())
            #print("label:", label.size())
            #print(label.size())
            topic_weight = torch.ones_like(label) + label * (label.size()[1]-1)

            #print(topic_weight)

            # Shape: (1,)
            loss = torch.nn.functional.binary_cross_entropy(probs, label.float(), topic_weight.float())

            pre_index_intent = convert((probs > 0.5).long(), self.vocab)
            pre_index = (probs > 0.5).long()

            label_intent = convert(label, self.vocab)
            #print("pre_index:", pre_index.size(), pre_index)
            #self.macro_f(pre_index.cpu(), label.cpu())
            '''total_pre = torch.sum(pre_index)
            total_true = torch.sum(label)
            mask_index = (label == 1).long()
            true_positive = (pre_index == label).long() * mask_index
            pre_true = torch.sum(true_positive)'''

            '''self.total_pre(total_pre.float().item())
            self.total_true(total_true.float().item())
            self.total_pre_true(pre_true.float().item())'''

            '''self.total_pre_macro(torch.sum(pre_index, dim=0))
            self.total_true_macro(torch.sum(label, dim=0))
            self.total_pre_true_macro(torch.sum(true_positive, dim=0))'''

            self.all_pre_intent(pre_index_intent.cpu())
            self.all_true_intent(label_intent.cpu())
            self.all_pre_intent_slot(pre_index.cpu())
            self.all_true_intent_slot(label.cpu())

            return {'loss': loss, 'probs': probs }

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        '''total_pre = self.total_pre.get_metric(reset=reset)
        total_pre_true = self.total_pre_true.get_metric(reset=reset)
        total_true = self.total_true.get_metric(reset=reset)'''

        '''total_pre_macro = self.total_pre_macro.get_metric(reset=reset)
        total_pre_true_macro = self.total_pre_true_macro.get_metric(reset=reset)
        total_true_macro = self.total_true_macro.get_metric(reset=reset)'''
        all_pre_intent = self.all_pre_intent.get_metric(reset=reset)
        all_true_intent = self.all_true_intent.get_metric(reset=reset)
        all_pre_intent_slot = self.all_pre_intent_slot.get_metric(reset=reset)
        all_true_intent_slot = self.all_true_intent_slot.get_metric(reset=reset)

        acc, rec, f1, facc = 0., 0., 0., 0.
        '''acc_macro = torch.zeros(total_pre_macro.size(0))
        rec_macro = torch.zeros(total_pre_macro.size(0))
        f1_macro_temp = torch.zeros(total_pre_macro.size(0))'''
        f1_macro = 0.
        '''if total_pre > 0:
            acc = total_pre_true / total_pre
        if total_true > 0:
            rec = total_pre_true / total_true
        if acc + rec > 0:
            f1 = 2 * acc * rec / (acc + rec)'''

        '''for x in range(total_pre_macro.size(0)):
            if total_pre_macro[x] == 0:
                acc_macro[x] = 0
            else:
                acc_macro[x] = total_pre_true_macro[x] / total_pre_macro[x]

        for x in range(total_pre_macro.size(0)):
            if total_true_macro[x] == 0:
                rec_macro[x] = 0
            else:
                rec_macro[x] = total_pre_true_macro[x] / total_true_macro[x]

        for x in range(total_pre_macro.size(0)):
            if acc_macro[x] + rec_macro[x] == 0:
                f1_macro_temp[x] = 0
            else:
                f1_macro_temp[x] = 2 * acc_macro[x] * rec_macro[x] / (acc_macro[x] + rec_macro[x])
        print("f1_macro_all:", f1_macro_temp)
        f1_macro = float(torch.sum(f1_macro_temp)/total_pre_macro.size(0))'''

        #print("all_true:", all_true)
        #print("all_pre:", all_pre)

        pre_micro_intent = precision_score(all_true_intent, all_pre_intent, average='micro')
        rec_micro_intent = recall_score(all_true_intent, all_pre_intent, average='micro')
        #acc_micro_intent = accuracy_score(all_true_intent, all_pre_intent)
        f1_micro_sk_intent = f1_score(all_true_intent, all_pre_intent, average='micro')
        f1_macro_sk_intent = f1_score(all_true_intent, all_pre_intent, average='macro')
        f1_weighted_intent = f1_score(all_true_intent, all_pre_intent, average='weighted')

        #metrics['acc_intent'] = acc_micro_intent
        metrics['pre_intent'] = pre_micro_intent
        metrics['rec_intent'] = rec_micro_intent

        #metrics['f1'] = f1
        #metrics['acc_macro'] = acc_macro
        #metrics['rec_macro'] = rec_macro
        #metrics['f1_macro'] = f1_macro
        metrics['f1_micro_sk_intent'] = f1_micro_sk_intent
        metrics['f1_macro_sk_intent'] = f1_macro_sk_intent
        metrics['f1_weighted_intent'] = f1_weighted_intent

        pre_micro_intent_slot = precision_score(all_true_intent_slot, all_pre_intent_slot, average='micro')
        rec_micro_intent_slot = recall_score(all_true_intent_slot, all_pre_intent_slot, average='micro')
        #acc_micro_intent_slot = accuracy_score(all_true_intent_slot, all_pre_intent_slot)
        f1_micro_sk_intent_slot = f1_score(all_true_intent_slot, all_pre_intent_slot, average='micro')
        f1_macro_sk_intent_slot = f1_score(all_true_intent_slot, all_pre_intent_slot, average='macro')
        f1_weighted_intent_slot = f1_score(all_true_intent_slot, all_pre_intent_slot, average='weighted')

        #metrics['acc_intent_slot'] = acc_micro_intent_slot
        metrics['pre_intent_slot'] = pre_micro_intent_slot
        metrics['rec_intent_slot'] = rec_micro_intent_slot

        # metrics['f1'] = f1
        # metrics['acc_macro'] = acc_macro
        # metrics['rec_macro'] = rec_macro
        # metrics['f1_macro'] = f1_macro
        metrics['f1_micro_sk_intent_slot'] = f1_micro_sk_intent_slot
        metrics['f1_macro_sk_intent_slot'] = f1_macro_sk_intent_slot
        metrics['f1_weighted_intent_slot'] = f1_weighted_intent_slot

        return metrics
