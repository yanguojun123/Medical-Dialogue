import json
import os
import Levenshtein as Le
import re
import argparse
import sys
def ArgsParser():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_number', default=10, type=int, required=True, help='split_number')

    return  parser.parse_args()


recommend_slot = ['medicine', 'treatment', 'precaution', 'medicine_category', 'medical_place', 'department']


def max_match(x, Dict):
    """

    :param x:
    :param Dict:
    :return:
    """
    ratio_max = 0
    entity = ''
    for y in Dict.keys():
        ratio = Le.ratio(x, y)
        if ratio > ratio_max:
            ratio_max = ratio
            entity = y
    return ratio_max, entity


def action_detection(sentence, role):
    """

    :param sentence:
    :param role:
    :return:
    """
    action_all=[]
    action = {}

    entity_all = json.load(open('entity_all.txt','r',encoding='utf-8'),strict=False)
    #value=re.sub('\s','',entity_all['disease']).split(',')
    #print(value)
    for slot in entity_all.keys():
        values = re.sub('\s','',entity_all[''+slot]).split(',')
        for value in values:
            if re.findall(value, sentence):
                action['slot'] = slot
                action['aspect'] = value
    ''' 
    disease_match = open('../../:disease_new.txt', 'r', encoding='utf-8').read().split(',')
    medicine_match = open('../../medicine_new.txt', 'r', encoding='utf-8').read().split(',')
    check_item_match = open('../../check_item_new.txt', 'r', encoding='utf-8').read().split(',')
    symptom_match = open('../../symptom_new.txt', 'r', encoding='utf-8').read().split(',')

    for disease in disease_match:
        if re.findall(disease, sentence):
            action['slot'] = 'disease'
            action['aspect'] = disease

    for symptom in symptom_match:
        if re.findall(symptom, sentence):
            action['slot'] = 'symptom'
            action['aspect'] = symptom

    for medicine in medicine_match:
        if re.findall(medicine, sentence):
            action['slot'] = 'disease'
            action['aspect'] = medicine

    for check_item in check_item_match:
        if re.findall(check_item, sentence):
            action['slot'] = 'check_item'
            action['aspect'] = check_item
    '''
    aa = '？|吗|啊？|呢？|么？|啥|会不会|怎么办'  # 询问
    bb = '().?()'  # 告知
    rem = '试试|可以口服|看看|可以吃|服用|涂抹|建议|需要|可以考虑'
    cc = '好的|谢谢|不客气|没事|ok|谢谢您|谢谢你'
    t1 = "(情况|症状|痛|发病|病|感觉|疼|这样|不舒服|大约).*?(多久|多长时间|几周了？|几天了？)"
    t2 = "(，|。|、|？)(多长时间了|多久了|有多久了|有多长时间了)|^(多久了|多长时间了|有多久了|有多长时间了|几天了|几周了)"
    t3 = "有多长时间|有多久"
    t4 = "([0-9]*|[一二三四五六七八九十几])(天|个月|个星期|周|星期|个多月)"
    dep = '科室|内科|男科|妇科|外科|心内科|泌尿科|中医科|营养科|皮肤性病科|耳鼻咽喉科|眼科|儿科|骨科'
    degree = "严重|不严重|轻微|加重|程度"
    fre = "([0-9]*|[一二三四五六七八九十几])(天一次)|每天几次"
    br = '哪个部位|哪个位置|哪里痛|什么部位|什么位置|哪个部位|哪个位置|哪一块|那个部位痛|描述一下位置|具体部位|具体位置'  # 身体部位
    dose = "ml|克|盒|瓶|个"
    pr = "sd"  # precaution
    effect = "不好|不大"
    dh = ""  # disease_history
    pa = ""  # pathogeny
    se = " "  # side_effect
    mc = ""  # medicine_category
    mp = "([一二三][甲乙丙]|[镇县区市省])医院"  # medicine_place
    tm = "体温|([0-9]*)度"  # temperature

    if re.search(aa, sentence):  # Inquire
        action['intent'] = 'Inquire'

    if role == 'doctor' and 'slot' in action.keys() and action['slot'] in recommend_slot:  # recommend
        action['intent'] = 'Recommend'

    if role == 'doctor' and 'slot' in action.keys() and action['slot'] == 'disease':  # diagnosis
        action['intent'] = 'Diagnosis'

    if re.search(cc, sentence):  # Chitchat
        action['intent'] = 'Chitchat'

    if re.search(t1, sentence) or re.search(t2, sentence) or re.search(t3, sentence):
        action['intent'] = 'Inquire'
        action['slot'] = 'time'

    if re.search(t4, sentence):
        action['slot'] = 'time'
        action['value'] = re.search(t4, sentence).group()

    if re.search(dep, sentence):
        action['slot'] = 'department'
        action['aspect'] = re.search(dep, sentence).group()

    if re.search(degree, sentence):
        action['slot'] = 'degree'
        action['value'] = re.search(degree, sentence).group()

    if re.search(fre, sentence):
        action['slot'] = 'frequency'
        action['value'] = re.search(fre, sentence).group()

    if re.search(br, sentence):
        action['slot'] = 'rang_body'
        action['aspect'] = re.search(br, sentence).group()

    if re.search(dose, sentence):
        action['slot'] = 'dose'
        action['value'] = re.search(dose, sentence).group()

    if re.search(mp, sentence):
        action['slot'] = 'medicine_place'
        action['aspect'] = re.search(mp, sentence).group()

    if re.search(tm, sentence):
        action['slot'] = 'temperature'
        action['value'] = re.search(tm, sentence).group()

    if 'intent' not in action.keys():
        action['intent'] = 'Inform'

    return action


def match_sentence(split_number):
    """

    :param split_number:
    :return:
    """
    result = []
    dialogue_labled_list = []
    dialogue_list = os.listdir('../dataset/')
    print('0-5000:', ','.join(dialogue_list[0:100]))
    print('15000-20000:', ','.join(dialogue_list[15000:15200]))
    #match_database = {}
    lable_data = json.load(open('Total_data.json', 'r', encoding='utf-8'))
    match_database = json.load(open('macth_database.json', 'r', encoding='utf-8'))
    dialogue_labled = 'pretrain_5000/Total_pretrained_data2_'

    '''
    disease_match=open('disease_new.txt', 'r', encoding='utf-8').read().split(',')
    medicine_match=open('medicine_new.txt', 'r', encoding='utf-8').read().split(',')
    check_item_macth=open('check_item_match.txt', 'r', encoding='utf-8').read().split(',')
    symptom_match=open('symptom_new.txt', 'r', encoding='utf-8').read().split(',')
    '''
    dialogue_match = json.load(open('dialogue_match.json', 'r', encoding='utf-8'))

    dialogue_labled_temp = os.listdir('label-data2/')  # Read the marked conversation IDs
    for x in dialogue_labled_temp:
        for y in os.listdir('label-data2/' + x + '/'):
            dialogue_labled_list.append(dialogue_match[re.findall(r'\d+', y)[0]+'.json'])
    print(len(dialogue_labled_list))

    '''dialogue_all = []
    for x in lable_data:
        information = x['information']
        for y in information:
            actions = y['actions']
            for z in actions:
                text = z['text']
                del z['text']
                del z['range']
                match_database['' + text] = z
    # print(match_database)
    json.dump(match_database, match_database_path, ensure_ascii=False)
    match_database_path.close()'''
    #total_dialogue numbers :95408
    start_index = split_number*5000
    end_index = (split_number+1)*5000 if split_number < 18 else 95408
    count = 0
    for x in dialogue_list[start_index:end_index]:
        #print(x)
        sys.stdout.flush()
        if x in dialogue_labled_list:
            print(x, 'find')
        if x not in dialogue_labled_list:
            path = '../dataset/'
            dialogue = json.load(open(path + x, 'r', encoding='utf-8'))['dialogues']
            # dialogue_labled=open('../../auto_labeled/labeled_'+x,'w',encoding='utf-8')
            temp_dialogue = {}
            temp_dialogue['dialogue'] = count
            temp_dialogue['information'] = dialogue
            for utterance in dialogue:
                sentence = utterance['sentence']
                role = utterance['role']
                del utterance['tokens']
                res = max_match(sentence, match_database)
                utterance['actions'] = []
                if res[0] > 0.8:  # find more matching sentences
                    utterance['actions'].append(match_database['' + res[1]])
                else:  # can not find more matching sentences
                    utterance['actions'].append(action_detection(sentence, role))
            count += 1
            result.append(temp_dialogue)
    with open('{}.json'.format(dialogue_labled + str(start_index)), 'wt', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=1)

args = ArgsParser()
match_sentence(args.split_number)
