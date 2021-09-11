import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import pickle
import Levenshtein as Le
#from dataset.translate import translateBaidu

import hashlib
import random
import openpyxl
from openpyxl import Workbook
import requests
import time
# set baidu develop parameter
apiurl = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
appid = '20210311000723421'
secretKey = 'sTccAEzixofaMEqax9p1'


class Getoutofloop(Exception):
  pass


key_map = {'symptom': ['临床症状及体征', '相关症状'], 'department': ['所属科室'], 'check_item':['检查', '影像学检查', '实验室检查'], 'body_range':'发病部位',
         'precuation': ['预防'], 'treatment': ['辅助治疗', '药物治疗', '手术治疗'], 'pathogeny': ['病因'], 'medicine': ['药物治疗', '治疗方案'],
         'medical_category': ['分类'], 'side_effect': ['不良反应']
         }
disease_entity = open('../data/Intermediate_file/disease_new.txt', 'r', encoding='utf-8').read().split(',')
symptom_entity = open('../data/Intermediate_file/symptom_new.txt', 'r', encoding='utf-8').read().split(',')
medicine_entity = open('../data/Intermediate_file/medicine_new.txt', 'r', encoding='utf-8').read().split(',')
check_item_entity = open('../data/Intermediate_file/check_item_new.txt', 'r', encoding='utf-8').read().split(',')
knowledge_entity = disease_entity + symptom_entity + medicine_entity + check_item_entity

entity = os.listdir('../data/knowledge_entities')


#get knowledge from the action
def action_knowledge_new(actions, state , role, knowledge_list):
    """
    Search knowledge
    :param actions:Action-Slot-Value
    :param state:Some entities recorded in the current conversation
    :param role:Dialogue role
    :param knowledge_list:List of knowledge to be matched
    :return:
    """
    triple = ['intent', 'slot', 'aspect', 'value']
    res = ''
    knowledge = []
    temp1 = []
    temp2 = []
    temp3 = []
    temp = ['', '', '']

    for action in actions:
         if 'slot' in action.keys() and 'aspect' in action.keys() and action['aspect'] in knowledge_entity:
            state.append(action['aspect'])
         if role == 'doctor' and 'slot' in action.keys() and action['slot'] in key_map.keys():
             if 'aspect' in action.keys():
                 temp1 += state
                 temp2 += key_map[action['slot']]
                 temp3.append(action['aspect'])
         for x in triple:
          if x in action.keys():
               res+=action[''+x]+' '
          else:
             res+='\'\''
         res=res[0:-1]
         res+=' <|continue|> '
    # Find all qualified knowledge triples from the knowledge list
    max_count = 0
    try:
        if role == 'doctor':
            for t1 in temp1:
                temp[0] = t1
                for t2 in temp2:
                    temp[1] = t2
                    for t3 in temp3:
                       temp[2] = t3
                       for slot in knowledge_list.keys():
                        if temp in knowledge_list[slot]:
                             knowledge.append(' '.join(temp))
                             max_count += 1
                             if max_count >= 5:
                                 raise Getoutofloop()
                             #print(temp)
            for t1 in temp1:
                temp[0] = t1
                for t2 in temp2:
                    temp[1] = t2
                    for slot in knowledge_list.keys():
                        for x in knowledge_list[slot]:
                            if temp[0] == x[0] and temp[1] == x[1]:
                                knowledge.append(' '.join(x))
                                max_count += 1
                                if max_count >= 5:
                                    raise Getoutofloop()
    except Getoutofloop:
        pass
    return res[:-14], '<|k|>'.join(set(knowledge))

def action_process_intent_slot(actions):#intent and slot
    triple=['intent','slot']
    res = ''
    for action in actions:
         for x in triple:
          if x in action.keys() :
               res += action[''+x]+' '
          else:
             res+='\'\''
         res=res[0:-1]
         res+='<|continue|>'
    return res[:-12]


def generate_K(original_data,result_path):
    """
    Sort out the manually marked dialogue
    :param original_data:origal_dialogue
    :param train_path:result_path
    :return:
    """
    #train_data=open('train.txt','w+',encoding='utf-8')
    #original_data=json.load(open('Total_data.json','r',encoding='utf-8'))
    #print(len(original_data))
    #random.shuffle(original_data)
    knowledge_list = json.load(open('../data/knowledge.json', 'r', encoding='utf-8'))

    for dialogue in original_data:
        context=dialogue['information']
        dialogue_state = []
        history = ''
        contact_text = ''
        contact_action = ''
        context_knowledge = ''
        user_intent=''
        user_current=''
        user_history=''
        for turn in range(len(context)):
            role = context[turn]['role']
            current_text = re.sub('[\n\r]', '', context[turn]['sentence'])
            current_action, knowledge = action_knowledge_new(context[turn]['actions'], dialogue_state, role, knowledge_list)
            #print(knowledge)
            # Check whether there are diseases, symptoms and other information
            if turn+1<len(context):
                next_role=context[turn+1]['role']
                contact_text+=current_text
                contact_action+=' <|continue|> '+current_action
                if knowledge != '':
                    context_knowledge += '<|k|>'+knowledge

                if role==next_role:
                    turn+=1
                    continue
                else:
                    if role == 'patient':
                       user_history = '<|user|> {} '.format(contact_text)
                       user_current = '<|currentuser|> {} <|endofcurrentuser|> '.format(contact_text)
                       user_intent = '<|intent|> {} <|endofintent|> '.format(contact_action[14:])# Remove the first '< continue >'
                       #history+=user_history
                    if role == 'doctor':
                        system_history = '<|system|> {}'.format(contact_text)
                        system_response = '<|response|> {} <|endofresponse|> '.format(contact_text)
                        system_action = '<|action|> {} <|endofaction|> '.format(contact_action[14:])
                        result_path.write('<|endoftext|> '+'<|context|> '+history+'<|endofcontext|> '+user_current+user_intent+' <|knowledge|> '+context_knowledge+' <|endofknowledge|> '+system_action+system_response+' <|endoftext|>\n')
                        history += user_history+system_history

                    contact_text=''
                    contact_action=''
                    context_knowledge = ''
            else:
                if role == 'patient':
                    user_history = '<|user|> {} '.format(contact_text)
                    user_intent = '<|intent|> {} <|endofintent|>'.format(contact_action)
                    history += user_history
                if role == 'doctor':
                    system_history = '<|system|> {}'.format(contact_text+current_text)
                    system_response = '<|response|> {} <|endofresponse|>'.format(contact_text+current_text)
                    system_action = '<|action|> {} <|endofaction|>'.format(contact_action+current_action[14:])
                    result_path.write('<|endoftext|> ' + '<|context|> '+history +'<|endofcontext|>'+ user_current+user_intent+' <|knowledge|> '+context_knowledge+' <|endofknowledge|> '+system_action + system_response + '<|endoftext|>\n')
                    history += system_history
                contact_text = ''
                contact_action = ''
                context_knowledge = ''
        result_path.write('\n')
    result_path.close()

def main():
    original_data = json.load(open('../data/Total_data_normalization.json', 'r', encoding='utf-8'))
    train_path = open('train_knowledge_num5.txt', 'w', encoding='utf-8')
    dev_path = open('dev_knowledge_num5.txt', 'w', encoding='utf-8')
    test_path = open('test_knowledge_num5.txt', 'w', encoding='utf-8')
    random.seed(5)
    random.shuffle(original_data)
    data_train = original_data[0:657]
    data_dev = original_data[657:757]
    data_test = original_data[757:1557]
    generate_K(data_train, train_path)
    generate_K(data_dev, dev_path)
    generate_K(data_test, test_path)
if __name__ == '__main__':
    main()