import json
import random

# argu = json.load(open('../data/argumentation_map.json', 'r', encoding='utf-8'))
#
# for key, value in argu.items():
#     key = key.replace('\n', '')

def get_data(origin_data, data_mapping, args):
    random.seed(args.seed)
    processed_data = []
    for data in origin_data:
        temp_processed_data = []
        temp_processed_data.append(data)
        knowledge = data.split('<|endofintent|>')[1].split('<|action|>')[0]
        temp_data = data.replace(' ', '')
        temp_data = temp_data.split('<|knowledge|>')[0]+temp_data.split('<|endofknowledge|>')[1]+'\n'
        if data_mapping[temp_data]['alias'] == [] and data_mapping[temp_data]['modify'] == []:
            argue_data_temp = data_mapping[temp_data]['trans'][0].replace('\n', '')
            argue_data = argue_data_temp.split('<|action|>')[0]+' '+knowledge+' '+argue_data_temp.split('<|endofintent|>')[1]
            temp_processed_data.append(argue_data)
        elif data_mapping[temp_data]['alias'] == []:
            argue_data_temp = data_mapping[temp_data]['modify'][0].replace('\n', '')
            argue_data = argue_data_temp.split('<|action|>')[0] + ' ' + knowledge + ' ' + \
                         argue_data_temp.split('<|endofintent|>')[1]
            temp_processed_data.append(argue_data)
        elif data_mapping[temp_data]['modify'] == []:
            argue_data_temp = data_mapping[temp_data]['alias'][0].replace('\n', '')
            argue_data = argue_data_temp.split('<|action|>')[0] + ' ' + knowledge + ' ' + \
                         argue_data_temp.split('<|endofintent|>')[1]
            temp_processed_data.append(argue_data)
        else:
            random_int = random.randint(0, 1)
            if random_int == 0:
                argue_data_temp = data_mapping[temp_data]['modify'][0].replace('\n', '')
                argue_data = argue_data_temp.split('<|action|>')[0] + ' ' + knowledge + ' ' + \
                             argue_data_temp.split('<|endofintent|>')[1]
                temp_processed_data.append(argue_data)
            else:
                argue_data_temp = data_mapping[temp_data]['alias'][0].replace('\n', '')
                argue_data = argue_data_temp.split('<|action|>')[0] + ' ' + knowledge + ' ' + \
                             argue_data_temp.split('<|endofintent|>')[1]
                temp_processed_data.append(argue_data)
        processed_data.append(temp_processed_data)
    return processed_data

# def get_data(origin_data, data_mapping, args):
#     random.seed(args.seed)
#     processed_data = []
#     for data in origin_data:
#         temp_data = data.replace(' ', '')
#         temp_data = temp_data.split('<|knowledge|>')[0] + temp_data.split('<|endofknowledge|>')[1] + '\n'
#         for key in ['alias', 'trans', 'modify']:
#             if data_mapping[temp_data][key] != []:
#                 for value in data_mapping[temp_data][key]:
#                     temp_processed_data = []
#                     temp_processed_data.append(data)
#                     temp_processed_data.append(value.replace('\n', ''))
#                     processed_data.append(temp_processed_data)
#     return processed_data

def get_new_np():
    data_argue = json.load(open('../data/argumentation_map.json', 'r', encoding='utf-8'))
    train_ha = open('../data/train_human_annotation.txt', 'r', encoding='utf-8')

    argue_np_train_new = open('../data/train_natural_perturbation_new.txt', 'w', encoding='utf-8')

    for dialogue in train_ha.read().split('\n\n')[0:-1]:
        for turn in dialogue.split('\n'):
            labels = turn.split('<|endofcurrentuser|>')[1].split('<|knowledge|>')[0] + turn.split('<|endofknowledge|>')[1]
            turn_temp = (turn.split('<|knowledge|>')[0]+turn.split('<|endofknowledge|>')[1]).replace(' ', '')+'\n'
            argue_turns = data_argue[turn_temp]
            for argue_type in argue_turns.keys():
                for argue_turn in argue_turns[argue_type]:
                    argue_new =argue_turn.split('<|intent|>')[0] + labels +'\n'
                    argue_np_train_new.write(argue_new)
        argue_np_train_new.write('\n')

    # for dialogue in dev_ha.read().split('\n\n')[0:-1]:
    #     for turn in dialogue.split('\n'):
    #         knowledge = turn.split('<|endofintent|>')[1].split('<|action|>')[0]
    #         turn_temp = turn.replace(' ', '')
    #         argue_turns = data_argue[turn_temp]
    #         for argue_type in argue_turns.keys():
    #             for argue_turn in argue_turns[argue_type]:
    #                 argue_new =argue_turn.split('<|knowledge|>')[0]+ ' '+knowledge + ' ' +argue_turn.split('<|endofknowledge|>')[1]+'\n'
    #                 argue_np_dev_new.write(argue_new)
    #     argue_np_dev_new.write('\n')
def get_new_dev():
    dev_ha = open('data/dev_human_annotation.txt', 'r', encoding='utf-8')
    dev_np = open('data/dev_natural_perturbation.txt', 'r', encoding='utf-8')
    argue_np_dev_new = open('../data/dev_natural_perturbation_new.txt', 'w', encoding='utf-8')



#get_new_np()