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

# Translated content source language translated language
def translateBaidu(content, fromLang='zh', toLang='en'):
    """
    Translate text from one specified language to another
    :param content:Content to be translated
    :param fromLang:source language
    :param toLang:target language
    :return:
    """
    #time.sleep(0.2)
    salt = str(random.randint(32768, 65536))
    sign = appid + content + salt + secretKey
    sign = hashlib.md5(sign.encode("utf-8")).hexdigest()
    try:
        paramas = {
            'appid': appid,
            'q': content,
            'from': fromLang,
            'to': toLang,
            'salt': salt,
            'sign': sign
        }
        response = requests.get(apiurl, paramas)
        jsonResponse = response.json()  #Get the returned result in JSON format
        dst = str(jsonResponse["trans_result"][0]["dst"])  # Get translated text results
        return dst
    except Exception as e:
        print(e)

key_map = {'symptom': ['临床症状及体征', '相关症状'], 'department': ['所属科室'], 'check_item':['检查', '影像学检查', '实验室检查'], 'body_range':'发病部位',
         'precuation': ['预防'], 'treatment': ['辅助治疗', '药物治疗', '手术治疗'], 'pathogeny': ['病因'], 'medicine': ['药物治疗', '治疗方案'],
         'medical_category': ['分类'], 'side_effect': ['不良反应']
         }

entity = os.listdir('../data/knowledge_entities')
chinese_tokens = open('ch_tokens.txt', 'r', encoding='utf-8').read().split('\n')

def random_modify(entity):
    """
    Random modification method
    :param entity:Entities to be modified
    :return:Modified entity
    """
    #random.seed(134)
    length = len(entity)
    m_entity = copy.deepcopy(entity)
    modify_index = random.randint(0, length-1)
    modify_type = random.randint(0,2)
    modify_tokens_index = random.randint(0,2669)
    if modify_type == 0 and modify_index+1 < length:#delete
        m_entity = entity[:modify_index]+entity[modify_index+1:]
    elif modify_type == 1 and modify_index+1 < length: #modify
        m_entity = entity[:modify_index]+chinese_tokens[modify_tokens_index]+entity[modify_index+1:]
    elif modify_type == 2: # add
        m_entity = entity[:modify_index]+chinese_tokens[modify_tokens_index]+entity[modify_index:]
    return m_entity
def max_match(x,List):
    """
    Calculate the maximum matching value in the list
    :param x:Entities to be calculated
    :param List:List to be matched
    :return:Maximum matching rate and corresponding entity
    """
    ratio_max=0
    entity=''
    for y in List:
        ratio=Le.ratio(x,y)
        if ratio>ratio_max:
            ratio_max=ratio
            entity=y
    return ratio_max,entity

def action_trans(actions):
    triple = ['intent', 'slot', 'aspect', 'value']
    res1 = ''
    res2 = ''
    for action in actions:
        for x in triple:
            if x in action.keys():
                if (x == 'aspect' or x == 'value') and action[x+''] !='' and action[x] is not None:
                    temp = translateBaidu(action[x])
                    if temp is not None:
                        temp = translateBaidu(temp, 'en', 'zh')
                    if temp is not None:
                        res2 += temp+ ' '
                else:
                    res2 += action['' + x] + ' '
                res1 += action['' + x] + ' '
            else:
                res1 += '\'\''
                res2 += '\'\''
        res1 = res1[0:-1]
        res2 = res2[0:-1]
        res1 += ' <|continue|> '
        res2 += ' <|continue|> '
    return res1[:-14], res2[:-14]

def action_process(actions):
    triple = ['intent', 'slot', 'aspect', 'value']
    res = ''
    for action in actions:
        for x in triple:
            if x in action.keys():
                res += action['' + x] + ' '
            else:
                res += '\'\''
        res = res[0:-1]
        res += ' <|continue|> '
    return res[:-14]

def action_alias(actions, medicine_alias): #medicine alias
    triple = ['intent', 'slot', 'aspect', 'value']
    res = ''
    for action in actions:
        if 'slot' in action.keys() and action['slot'] == 'medicine' and 'aspect' in action.keys() and action['aspect']+'.txt' in entity:
            medicine = json.load(open('../data/knowledge_entities/' + action['aspect'] + '.txt', 'r', encoding='utf-8'))
            for link in medicine['links']:
                if link[1] in ['商品名', '英文名称', '通用名']:
                    if action['aspect'] in medicine_alias.keys():
                        if action['aspect'] == link[2]:
                            link[2] = []
                        elif action['aspect'] in link[2]:
                            #print(link[2])
                            link[2].remove(action['aspect'])
                        medicine_alias[''+action['aspect']] |= set(link[2])
                    else:
                        medicine_alias[''+action['aspect']] = set()
        for x in triple:
            if x in action.keys():
                res += action['' + x] + ' '
            else:
                res += '\'\''
        res = res[0:-1]
        res += ' <|continue|> '
    return res[:-14]

def action_modify(actions):
    triple = ['intent', 'slot', 'aspect', 'value']
    res = ''
    temp = {}
    for action in actions:
        if 'aspect' in action.keys() and action['aspect']!='':
            temp[''+action['aspect']] = random_modify(action['aspect'])
        for x in triple:
            if x in action.keys():
                res += action['' + x] + ' '
            else:
                res += '\'\''
        res = res[0:-1]
        res += ' <|continue|> '
    return res[:-14],temp


def generate_modify(original_data,result_path):
    """
    Generate new data by randomly modifying the method
    :param original_data:original_data
    :param train_path:result_path
    :return:
    """
    #train_data=open('train.txt','w+',encoding='utf-8')
    #original_data=json.load(open('Total_data.json','r',encoding='utf-8'))
    #print(len(original_data))
    #random.shuffle(original_data)
    #knowledge_list = json.load(open('../../../knowledge.json', 'r', encoding='utf-8'))

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
            #current_action, knowledge = action_knowledge_new(context[turn]['actions'], dialogue_state, role,
                                                             #knowledge_list)
            current_action, modify_entity = action_modify(context[turn]['actions'])
            #current_action = action_process(context[turn]['actions'])

            print(modify_entity)
            # Check whether there are diseases, symptoms and other information
            if turn+1<len(context):
                next_role=context[turn+1]['role']
                contact_text+=current_text
                contact_action+=' <|continue|> '+current_action
                '''if knowledge != '':
                    context_knowledge += '<|k|>'+knowledge'''
                if role==next_role: #The roles of the two sentences are the same, merging
                    turn+=1
                    continue
                else:
                    if role == 'patient':
                       user_history = '<|user|> {} '.format(contact_text)
                       user_current = '<|currentuser|> {} <|endofcurrentuser|> '.format(contact_text)
                       user_intent = '<|intent|> {} <|endofintent|> '.format(contact_action[14:])# 去掉最前面的一个'<continue>'
                       #history+=user_history
                    if role == 'doctor':
                        system_history = '<|system|> {}'.format(contact_text)
                        system_response = '<|response|> {} <|endofresponse|> '.format(contact_text)
                        system_action = '<|action|> {} <|endofaction|> '.format(contact_action[14:])
                        '''res = '<|endoftext|> '+'<|context|> '+history+'<|endofcontext|> '+user_current+user_intent\
                              +' <|knowledge|> '+context_knowledge+' <|endofknowledge|> '+system_action+system_response+' <|endoftext|>\n'''
                        res = '<|endoftext|> ' + '<|context|> ' + history + '<|endofcontext|> ' + user_current + user_intent \
                              + system_action + system_response + ' <|endoftext|>\n'
                        #train_path.write(res)
                        for entity in modify_entity.keys(): #Add new modified data
                            if entity in '<|endoftext|> ' + '<|context|> ' + history + '<|endofcontext|> ' + user_current:
                                temp_res = re.sub(entity, modify_entity[entity],'<|endoftext|> ' + '<|context|> '
                                                  + history + '<|endofcontext|> ' + user_current)+user_intent+\
                                           system_action+system_response+' <|endoftext|>\n'
                                result_path.write(temp_res)
                        history += user_history+system_history
                    contact_text=''
                    contact_action=''
            else: #last sentence
                if role == 'patient':
                    user_history = '<|user|> {} '.format(contact_text)
                    user_intent = '<|intent|> {} <|endofintent|>'.format(contact_action)
                    history += user_history
                if role == 'doctor':
                    system_history = '<|system|> {}'.format(contact_text+current_text)
                    system_response = '<|response|> {} <|endofresponse|>'.format(contact_text+current_text)
                    system_action = '<|action|> {} <|endofaction|>'.format(contact_action+current_action[14:])
                    res = '<|endoftext|> ' + '<|context|> ' + history + '<|endofcontext|> ' + user_current + user_intent\
                          + system_action + system_response + ' <|endoftext|>\n'
                    #train_path.write(res)
                    for entity in modify_entity.keys():  #Add new modified data
                        if entity in '<|endoftext|> ' + '<|context|> ' + history + '<|endofcontext|> ' + user_current:
                            temp_res = re.sub(entity,modify_entity[entity],'<|endoftext|> ' + '<|context|> ' + history + '<|endofcontext|> ' + user_current) \
                                       + user_intent + system_action + system_response + ' <|endoftext|>\n'
                            result_path.write(temp_res)
                    history += system_history
                contact_text = ''
                contact_action = ''
        result_path.write('\n')
    result_path.close()

def generate_trans(original_data, result_path):
    """
    The back translation method generates new data
    :param original_data:Original_data
    :param result_path:Result_path
    :return:
    """
    #knowledge_list = json.load(open('../../../knowledge.json', 'r', encoding='utf-8'))
    for dialogue in original_data:
        context = dialogue['information']
        history = ''
        history_trans = ''
        contact_text = ''
        contact_text_trans = ''
        contact_action = ''
        user_intent = ''
        user_intent_trans = ''
        user_current = ''
        user_current_trans = ''
        user_history = ''
        user_history_trans = ''
        contact_action_trans = ''
        turn = 0
        max_turn = len(context)
        dialogue_state = []
        for turn in range(len(context)):
            role = context[turn]['role']
            current_text = re.sub('[\n\r]', '', context[turn]['sentence'])
            current_text_trans = ''
            if current_text != '' and current_text is not None:
                temp = translateBaidu(current_text)
                if temp != '' and temp is not None:
                    current_text_trans = translateBaidu(temp, 'en', 'zh')
            '''current_action, knowledge = action_knowledge_new(context[turn]['actions'], dialogue_state, role,
                                                             knowledge_list)'''
            current_action, current_action_trans = action_trans(context[turn]['actions'])
            if turn + 1 < len(context):
                next_role = context[turn + 1]['role']
                contact_text += current_text
                if current_text_trans is not None:
                    contact_text_trans += current_text_trans
                contact_action += ' <|continue|> ' + current_action
                if current_action_trans is not None:
                    contact_action_trans += ' <|continue|> ' + current_action_trans

                if role == next_role:
                    turn += 1
                    continue
                else:
                    if role == 'patient':
                        user_history = '<|user|> {} '.format(contact_text)
                        user_current = '<|currentuser|> {} <|endofcurrentuser|> '.format(contact_text)
                        user_intent = '<|intent|> {} <|endofintent|> '.format(
                            contact_action[14:])  # Remove the first '< continue >'
                        # history+=user_history
                        # trans
                        user_history_trans = '<|user|> {} '.format(contact_text_trans)
                        user_current_trans = '<|currentuser|> {} <|endofcurrentuser|> '.format(contact_text_trans)
                        user_intent_trans = '<|intent|> {} <|endofintent|> '.format(
                            contact_action_trans[14:])  # Remove the first '< continue >'

                    if role == 'doctor':
                        system_history = '<|system|> {}'.format(contact_text)
                        system_response = '<|response|> {} <|endofresponse|> '.format(contact_text)
                        system_action = '<|action|> {} <|endofaction|> '.format(contact_action[14:])
                        temp_res ='<|endoftext|> ' + '<|context|> ' + history + '<|endofcontext|> ' + user_current +\
                                  user_intent +system_action + system_response + ' <|endoftext|>\n'

                        # trans
                        system_history_trans = '<|system|> {}'.format(contact_text_trans)
                        system_response_trans = '<|response|> {} <|endofresponse|> '.format(contact_text_trans)
                        system_action_trans = '<|action|> {} <|endofaction|> '.format(contact_action_trans[14:])
                        temp_res_trans = '<|endoftext|> ' + '<|context|> ' + history_trans + '<|endofcontext|> ' + user_current_trans \
                                         + user_intent_trans + system_action_trans + system_response_trans + ' <|endoftext|>\n'
                        #result_path.write(temp_res)
                        if temp_res_trans != temp_res:
                            result_path.write(temp_res_trans)
                        history += user_history + system_history
                        history_trans += user_history_trans + system_history_trans
                    contact_text = ''
                    contact_action = ''
                    contact_action_trans = ''
                    contact_text_trans = ''

            else:
                    if role == 'patient':
                        user_history = '<|user|> {} '.format(contact_text)
                        user_intent = '<|intent|> {} <|endofintent|>'.format(contact_action)
                        history += user_history
                        #trans
                        contact_text_trans = '' if contact_text_trans is None else contact_text_trans
                        contact_action_trans = '' if contact_action_trans is None else contact_action_trans
                        user_history_trans = '<|user|> {} '.format(contact_text_trans)
                        user_intent_trans = '<|intent|> {} <|endofintent|>'.format(contact_action_trans)
                        history_trans += user_history_trans
                    if role == 'doctor':
                        system_history = '<|system|> {}'.format(contact_text + current_text)
                        system_response = '<|response|> {} <|endofresponse|>'.format(contact_text + current_text)
                        system_action = '<|action|> {} <|endofaction|>'.format(contact_action + current_action[14:])
                        temp_res = '<|endoftext|> ' + '<|context|> ' + history + '<|endofcontext|> ' + user_current + user_intent \
                                   + system_action + system_response + ' <|endoftext|>\n'
                        #result_path.write(temp_res)
                        history += system_history
                        #trans
                        contact_text_trans = '' if contact_text_trans is None else contact_text_trans
                        contact_action_trans = '' if contact_action_trans is None else contact_action_trans
                        current_text_trans = '' if current_text_trans is None else current_text_trans
                        current_action_trans = '' if current_action_trans is None else current_action_trans
                        system_history_trans = '<|system|> {}'.format(contact_text_trans + current_text_trans)
                        system_response_trans = '<|response|> {} <|endofresponse|>'.format(contact_text_trans + current_text_trans)
                        system_action_trans = '<|action|> {} <|endofaction|>'.format(contact_action_trans + current_action_trans[14:])
                        temp_res_trans = '<|endoftext|> ' + '<|context|> ' + history_trans + '<|endofcontext|> ' + user_current_trans + user_intent_trans+\
                                          system_action_trans + system_response_trans + ' <|endoftext|>\n'
                        if temp_res_trans != temp_res:# if different and add
                            result_path.write(temp_res_trans)
                        history_trans += system_history_trans
                    contact_text = ''
                    contact_action = ''
                    contact_text_trans = ''
                    contact_action_trans = ''
                    context_knowledge = ''
        result_path.write('\n')
    result_path.close()

def generate_alias(original_data, result_path):
    """
    The alias method generates new data
    :param original_data:Original_data
    :param result_path:Result_path
    :return:
    """
    alias = ['商品名', '英文名称', '通用名']

    #knowledge_list = json.load(open('../../../knowledge.json', 'r', encoding='utf-8'))
    for dialogue in original_data:  # 外层大循环
        dialogue_state = []
        context = dialogue['information']
        history = ''
        contact_text = ''
        contact_action = ''
        user_intent = ''
        user_current = ''
        user_history = ''
        knowledge = ''
        context_knowledge = ''
        medicine = {}
        for turn in range(len(context)):
            role = context[turn]['role']
            current_text = re.sub('[\n\r]', '', context[turn]['sentence'])
            #current_action, knowledge = action_knowledge_new(context[turn]['actions'], dialogue_state, role,
            #                                                 knowledge_list)
            current_action = action_alias(context[turn]['actions'], medicine)
            if len(medicine) != 0:
                print(medicine)
            if turn + 1 < len(context):
                next_role = context[turn + 1]['role']
                contact_text += current_text
                contact_action += '<|continue|>' + current_action
                #print(contact_action)
                if knowledge != '':
                    context_knowledge += '<|k|>'+knowledge
                if role == next_role:
                    turn += 1
                    continue
                else:
                    if role == 'patient':
                        user_history = '<|user|> {} '.format(contact_text)
                        user_current = '<|currentuser|> {} <|endofcurrentuser|> '.format(contact_text)
                        user_intent = '<|intent|> {} <|endofintent|> '.format(
                            contact_action[12:])  # Remove the first '< continue >'
                        # history+=user_history
                    if role == 'doctor':
                        system_history = '<|system|> {}'.format(contact_text)
                        system_response = '<|response|> {} <|endofresponse|> '.format(contact_text)
                        system_action = '<|action|> {} <|endofaction|> '.format(contact_action[12:])
                        temp_res ='<|endoftext|> ' + '<|context|> ' + history + '<|endofcontext|> ' + user_current
                        #train_path.write(temp_res)
                        for medicine_keys in medicine.keys():
                            for alia in medicine[medicine_keys]:
                                #print(alia)
                                if ''+medicine_keys in temp_res:
                                    temp = re.sub('' + medicine_keys, alia, temp_res)+ user_intent \
                                  + system_action + system_response + ' <|endoftext|>\n'
                                    if temp != '\n':
                                        result_path.write(temp)
                        history += user_history + system_history
                    contact_text = ''
                    contact_action = ''
                    context_knowledge = ''
            else:
                    if role == 'patient':
                        user_history = '<|user|> {} '.format(contact_text)
                        user_intent = '<|intent|> {} <|endofintent|>'.format(contact_action)
                        history += user_history
                    if role == 'doctor':
                        system_history = '<|system|> {}'.format(contact_text + current_text)
                        system_response = '<|response|> {} <|endofresponse|>'.format(contact_text + current_text)
                        system_action = '<|action|> {} <|endofaction|>'.format(contact_action + current_action[14:])
                        temp_res = '<|endoftext|> ' + '<|context|> ' + history + '<|endofcontext|> ' + user_current
                        #train_path.write(temp_res)
                        for medicine_keys in medicine.keys():
                            for alia in medicine[medicine_keys]:
                                if '' + medicine_keys in temp_res:
                                    temp = re.sub('' + medicine_keys, alia, temp_res) + user_intent \
                                           + system_action + system_response + ' <|endoftext|>\n'
                                    if temp != '\n':
                                        result_path.write(temp)
                        history += system_history
                    contact_text = ''
                    contact_action = ''
                    context_knowledge = ''
        #train_path.write('\n')
    result_path.close()

def main():
    """
    Main function
    :return:
    """
    original_data = json.load(open('../data/Total_data_normalization.json', 'r', encoding='utf-8'))
    train_path = open('../data/train_natural_perturbation.txt', 'a', encoding='utf-8')
    dev_path = open('../data/dev_natural_perturbation.txt', 'a', encoding='utf-8')


    random.seed(5)
    random.shuffle(original_data)
    data_train = original_data[0:657]
    data_dev = original_data[657:757]
    data_test = original_data[757:1557]

    generate_trans(data_train, train_path)
    generate_trans(data_dev, dev_path)

    generate_alias(data_train, train_path)
    generate_alias(data_dev, dev_path)

    generate_modify(data_train, train_path)
    generate_modify(data_dev, dev_path)
if __name__ == '__main__':
    main()