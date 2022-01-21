from nlgeval import compute_metrics
import nltk
from nltk.translate.meteor_score import *
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
# nlgeval = NLGEval()
from common.args import ArgsParser
import numpy as np
import json
import re
from sklearn.metrics import f1_score, recall_score, precision_score
from typing import Iterable, Tuple, Dict, Set, List
from allennlp.training.metrics.metric import Metric
import copy


class NLTK_BLEU(Metric):
    def __init__(
            self,
            smoothfunc: SmoothingFunction = None,
            ngram_weights: Iterable[float] = (0.25, 0.25, 0.25, 0.25),
            #file_path: str= ''
    ) -> None:
        self._ngram_weights = ngram_weights
        self._scores = []
        #self.smoothfunc = SmoothingFunction().method7
        self.smoothfunc = smoothfunc
        #self.file_path = open('bleu_test/'+file_path,'w',encoding='utf-8')
        # if all(ngram_weights = SmoothingFunction().method0

    def reset(self) -> None:
        self._scores = []

    # @overrides
    def get_metric(self, reset: bool = False):
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
args = ArgsParser().parse()
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
    :param string:Generated string
    :return:
    """
    string = re.sub('\'\'', '', string)
    for s in string:
        if u'\u4e00' <= s <= u'\u9fff':
            return s
    return 'F'

def split_chinese(str):
    """
    Split Chinese
    :param str:Processed string
    :return:Chinese string, connected with a space in the middle
    """
    temp1=str.split(' ')
    res=[]
    for x in temp1:
        if exist_chinese(x):
            temp2=list(x)
            res+=temp2
        #res.append(x)
    return ''.join(res)

def intent_evaluation():
    """
    Evaluate NLU
    :return:
    """
    y_pred_intent = []
    y_gt_intent = []
    y_pred_intent_slot = []
    y_gt_intent_slot = []
    disease_pred = []
    disease_gt = []
    success_count = 0
    count = 0  # total intent number
    total_action = 0
    true_action = 0
    result_path = args.evaluation_path
    result = json.load(open(result_path, 'r', encoding='utf-8'))
    hyp_intent = open('hyp_intent.txt', 'w', encoding='utf-8')
    ref_intent = open('ref_intent.txt', 'w', encoding='utf-8')
    hyp_intent_list=[]
    ref_intent_list=[]
    hyp_value_list=[]
    ref_value_list=[]

    for x in result.keys():

        # print(len(result[''+str(x)]['target_intent']))
        # print(len(result['' + str(x)]['generated_intent']))
        # print(result[''+str(x)]['target_intent'])
        # print(result['' + str(x)]['generated_intent'])
        #print(len(result[x]['generated_intent']),len(result[x]['target_intent']))
        for y in range(min(len(result[x]['generated_intent']), len(result[x]['target_intent']))):

            #print("sdsds")
            y_pred_intent_temp = [0] * 5
            y_gt_intent_temp = [0] * 5
            y_pred_intent_slot_temp = [0] * len(intent_slot1.keys())
            y_gt_intent_slot_temp = [0] * len(intent_slot1.keys())

            # Split the actions of the ground truth
            ref_intent_temp = result[x]['target_intent'][y]
            #print("ref_intent_temp:",ref_intent_temp)
            ref_intent_temp = re.sub('\'\'', '', ref_intent_temp)

            ref_delete_english=re.sub('[\sa-zA-Z<>|_]','',ref_intent_temp)
            if ref_delete_english!='':
                ref_value_list.append(' '.join(list(split_chinese(ref_delete_english))))
                #print('ref_delete_english:',split_chinese(ref_delete_english))

            ref_temp=re.sub('<\|continue\|>','',ref_intent_temp)
            ref_intent_list.append(split_chinese(ref_temp))
            ref_intent_temp = ref_intent_temp.split('<|continue|>')
            ref_intent.write(''.join(ref_intent_temp) + '\n')

            total_action += len(ref_intent_temp)

            for i in ref_intent_temp:  # Change the corresponding position to 1
                i = re.sub('\'\'', '', i)
                if exist_chinese(i) != 'F' and (i.split(exist_chinese(i))[0]).strip() in intent_slot1.keys():
                    y_gt_intent_slot_temp[int(intent_slot1['' + (i.split(exist_chinese(i))[0]).strip()])] = 1
                if exist_chinese(i) == 'F' and i.strip() in intent_slot1.keys():
                    y_gt_intent_slot_temp[int(intent_slot1['' + i.strip()])] = 1
                # if exist_chinese(i)!='F':

                if i.strip().split(' ')[0] in intent1.keys():
                    y_gt_intent_temp[int(intent1['' + i.strip().split(' ')[0]])] = 1
                if i.strip() in intent1.keys():
                    y_gt_intent_temp[int(intent1['' + i.strip()])] = 1

            #Process generated actions
            generate_intent = pattern.sub(r'\1\2', result[x]['generated_intent'][y][1:-1])
            generate_intent = pattern.sub(r'\1\2', generate_intent)


            hyp_delete_english = re.sub('[\sa-zA-Z]', '', generate_intent)
            if ref_delete_english != '':
                hyp_value_list.append(' '.join(list(split_chinese(hyp_delete_english))))

            hyp_temp = re.sub('<\|continue\|>', '', generate_intent)
            hyp_intent_list.append(split_chinese(hyp_temp))
            generate_intent_temp = generate_intent.split('<|continue|>')
            # generate_intent_temp=result[x]['generated_intent'][y].split('<|continue|>')
            hyp_intent.write(''.join(generate_intent_temp) + '\n')

            for i in generate_intent_temp:  # Change the corresponding position to 1
                if i in ref_intent_temp:
                    true_action += 1
                i = re.sub('\'\'', '', i)
                if exist_chinese(i) != 'F' and (i.split(exist_chinese(i))[0]).strip() in intent_slot1.keys():
                    y_pred_intent_slot_temp[int(intent_slot1['' + (i.split(exist_chinese(i))[0]).strip()])] = 1
                if exist_chinese(i) == 'F'and i.strip() in intent_slot1.keys():
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
    ref_intent.close()
    hyp_intent.close()
    y_pred_intent = np.array(y_pred_intent)
    y_gt_intent = np.array(y_gt_intent)
    y_gt_intent_slot = np.array(y_gt_intent_slot)
    y_pred_intent_slot = np.array(y_pred_intent_slot)

    bleu1 = NLTK_BLEU(ngram_weights=(1, 0, 0, 0), smoothfunc=SmoothingFunction().method0)
    bleu4 = NLTK_BLEU(ngram_weights=(0, 0, 0, 1), smoothfunc=SmoothingFunction().method0)
    bleu1_value = NLTK_BLEU(ngram_weights=(1, 0, 0, 0), smoothfunc=SmoothingFunction().method0)
    bleu1(ref_intent_list, hyp_intent_list)
    bleu4(ref_intent_list, hyp_intent_list)
    bleu1_value(ref_value_list, hyp_value_list)
    print("pre_intent:", y_pred_intent.shape)
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
    #print("intent ratio:", np.sum(y_gt_intent, axis=0)/np.sum(y_gt_intent))
    #metric['NLU']['intent_pre'] = precision_score(y_gt_intent, y_pred_intent, average='micro')
    #metric['NLU']['intent_rec'] = recall_score(y_gt_intent, y_pred_intent, average='micro')
    #metric['NLU']['intent_pre1'] = precision_score(y_gt_intent, y_pred_intent, average='macro')
    #metric['NLU']['intent_rec1'] = recall_score(y_gt_intent, y_pred_intent, average='macro')
    metric['NLU']['intent_weighted'] = f1_score(y_gt_intent, y_pred_intent, average='weighted')
    metric['NLU']['intent_macro'] = f1_score(y_gt_intent, y_pred_intent, average='macro')
    metric['NLU']['intent_micro'] = f1_score(y_gt_intent, y_pred_intent, average='micro')
    #metric['NLU']['intent_slot_pre'] = precision_score(y_gt_intent_slot, y_pred_intent_slot, average='micro')
    #metric['NLU']['intent_slot_rec'] = recall_score(y_gt_intent_slot, y_pred_intent_slot, average='micro')

    metric['NLU']['intent_slot_macro_weighted'] = f1_score(y_gt_intent_slot, y_pred_intent_slot, average='weighted')
    metric['NLU']['intent_slot_macro'] = f1_score(y_gt_intent_slot, y_pred_intent_slot, average='macro')
    metric['NLU']['intent_slot_micro'] = f1_score(y_gt_intent_slot, y_pred_intent_slot, average='micro')
    metric['NLU']['bleu1'] = bleu1_value.get_metric(reset=False)
    #metric['NLU']['bleu1_value'] = bleu1_value.get_metric(reset=False)
    #metric['NLU']['accuracy'] = true_action / total_action
    #metric['NLU']['bleu4'] = bleu4.get_metric(reset=False)
    metric['NLU']['combine'] = (metric['NLU']['intent_slot_micro']*0.5)+(0.5*metric['NLU']['bleu1'])
    #print(metric)
    #metrics_dict = compute_metrics(references=['ref_intent.txt'], hypothesis='hyp_intent.txt')
    print(metric)
    #metric['NLU'].update(metrics_dict)


def action_evaluation():
    """
    Evaluate DPL task
    :return:
    """
    y_pred_intent = []
    y_gt_intent = []
    y_pred_intent_slot = []
    y_gt_intent_slot = []
    total_action = 0
    true_action = 0  # total intent number
    result_path = args.evaluation_path
    result = json.load(open(result_path, 'r', encoding='utf-8'))
    hyp_intent = open('hyp_action.txt', 'w', encoding='utf-8')
    ref_intent = open('ref_action.txt', 'w', encoding='utf-8')

    hyp_intent_list = []
    ref_intent_list = []
    hyp_value_list = []
    ref_value_list = []

    for x in result.keys():
        # print(len(result[''+str(x)]['target_intent']))
        # print(len(result['' + str(x)]['generated_intent']))
        # print(result[''+str(x)]['target_intent'])
        # print(result['' + str(x)]['generated_intent'])
        if len(result[x]['generated_action']) > 0:
            for y in range(min(len(result[x]['generated_action']), len(result[x]['target_action']))):
                y_pred_intent_temp = [0] * 7
                y_gt_intent_temp = [0] * 7
                y_pred_intent_slot_temp = [0] * 56
                y_gt_intent_slot_temp = [0] * 56
                ref_intent_temp = result[x]['target_action'][y]
                ref_intent_temp = re.sub('\'\'', '', ref_intent_temp)

                #print('ref_intent_temp:',ref_intent_temp)
                ref_delete_english = re.sub('[\sa-zA-Z<|>，_\']', '', ref_intent_temp)
                if ref_delete_english != '':
                    #print('ref:',split_chinese(ref_delete_english))
                    ref_value_list.append(' '.join(list(split_chinese(ref_delete_english))))

                ref_temp = re.sub('<\|continue\|>', '', ref_intent_temp)
                ref_intent_list.append(split_chinese(ref_temp))
                ref_intent_temp = ref_intent_temp.split('<|continue|>')
                ref_intent.write(''.join(ref_intent_temp) + '\n')

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
                generate_intent = pattern.sub(r'\1\2', result[x]['generated_action'][y][1:-1])
                generate_intent = pattern.sub(r'\1\2', generate_intent)

                hyp_delete_english = re.sub('[\sa-zA-Z<|>，_\']', '', generate_intent)
                if ref_delete_english != '':
                    #print('hyp:', split_chinese(hyp_delete_english))
                    hyp_value_list.append(' '.join(list(split_chinese(hyp_delete_english))))

                hyp_temp = re.sub('<\|continue\|>', '', generate_intent)
                hyp_intent_list.append(split_chinese(hyp_temp))

                generate_intent_temp = generate_intent.split('<|continue|>')
                # generate_intent_temp = result[x]['generated_action'][y].split('<|continue|>')
                hyp_intent.write(''.join(generate_intent_temp) + '\n')

                for i in generate_intent_temp:  # Change the corresponding position to 1
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

    ref_intent.close()
    hyp_intent.close()
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


    metric['AP']['intent_weighted'] = f1_score(y_gt_intent, y_pred_intent, average='weighted')
    metric['AP']['intent_macro'] = f1_score(y_gt_intent, y_pred_intent, average='macro')
    metric['AP']['intent_micro'] = f1_score(y_gt_intent, y_pred_intent, average='micro')
    metric['AP']['intent_slot_weighted'] = f1_score(y_gt_intent_slot, y_pred_intent_slot, average='weighted')
    metric['AP']['intent_slot_macro'] = f1_score(y_gt_intent_slot, y_pred_intent_slot, average='macro')
    metric['AP']['intent_slot_micro'] = f1_score(y_gt_intent_slot, y_pred_intent_slot, average='micro')
    metric['AP']['bleu1'] = bleu1_value.get_metric(reset=False)
    metric['AP']['accuracy'] = true_action / total_action
    #metric['AP']['bleu4'] = bleu4.get_metric(reset=False)
    metric['AP']['combined'] = (metric['AP']['intent_slot_micro'] * 0.5) + (0.5 * metric['AP']['bleu1'])
    print(metric)
    #metrics_dict = compute_metrics(references=['ref_action.txt'], hypothesis='hyp_action.txt')
    # print(f1_score(y_gt_intent_slot, y_pred_intent_slot, average=None))


def generate_text():
    """
    Evaluate response
    :return:
    """
    result_path = args.evaluation_path
    result = json.load(open(result_path, 'r', encoding='utf-8'))
    hyp = open('hyp.txt', 'w', encoding='utf-8')
    ref = open('ref.txt', 'w', encoding='utf-8')
    hyp_list = []
    hyp_list_1 = []
    ref_list = []
    total_vocabs1 = 0
    appear_vocabs1 = set()
    for x in result:
        # for y in range(len(result[x]['target_response'])):
        for y in range(len(result[x]['generated_response'])):
            # hyp_list.append(result[x]['target_response'][y])
            # ref_list.append(result[x]['generated_response'][y])
            if y >= len(result[x]['generated_response']):
                hyp.write('\n')
            else:
                # hyp.write(re.sub(' ', '', result[x]['generated_response'][y]) + '\n')
                hyp.write(result[x]['generated_response'][y] + '\n')
            # hyp.write(re.sub(' ', '', result[x]['generated_response'][y]) + '\n')
            hyp_list_1.append(re.sub(' ', '', result[x]['generated_response'][y]))
            #hyp_list_1.append(result[x]['generated_response'][y].split(' '))
            #hyp_list.append(result[x]['generated_response'][y])
            hyp_list.append(' '.join(list(result[x]['generated_response'][y])))
            # total_vocabs1+=len(re.sub(' ', '', result[x]['generated_response'][y]))
            # appear_vocabs1.update(re.sub(' ', '', result[x]['generated_response'][y]))
            # ref.write(re.sub('\s','',result[x]['target_response'][y])+'\n')
            ref.write(' '.join(list(result[x]['target_response'][y])) + '\n')
            ref_list.append(' '.join(list(result[x]['target_response'][y])))

        # ref.write('\n')
        # hyp.write('\n')
    ref.close()
    hyp.close()
    metric['RG']['distinct1'] = distinct1(hyp_list_1)
    metric['RG']['distinct2'] = distinct2(hyp_list_1)
    print(metric)

    meteor_result = 0
    count = 0
    for index in range(len(ref_list)):
        if len(ref_list[index].split(' ')) != 0 and len(hyp_list[index].split(' ')) != 0:
            res = round(meteor_score([ref_list[index]], hyp_list[index]), 1)
            meteor_result += res
            count += 1


    print("meteor_score", meteor_result / count)

    #compute Rouge
    rouge_1 = 0
    for index in range(len(ref_list)):
        if len(ref_list[index].split(' ')) != 0 and len(hyp_list[index].split(' ')) != 0:
            grams_reference = list(ref_list[index])
            grams_model = list(hyp_list[index])
            temp = 0
            ngram_all = len(grams_reference)
            for x in grams_reference:
                if x in grams_model: temp = temp + 1
            rouge_1 += temp / ngram_all
    print("rouge_1", rouge_1 / count)

    smoothfunc = [None, SmoothingFunction().method0, SmoothingFunction().method1, SmoothingFunction().method2,
                  SmoothingFunction().method5, SmoothingFunction().method7]
    for index, item in enumerate(smoothfunc):
        bleu1 = NLTK_BLEU(ngram_weights=(1, 0, 0, 0),smoothfunc=item)
        # bleu2 = NLTK_BLEU(ngram_weights=(0, 1, 0, 0),smoothfunc=item,file_path=str(index))
        bleu4 = NLTK_BLEU(ngram_weights=(0, 0, 0, 1), smoothfunc=item)
        bleu_aver = NLTK_BLEU(ngram_weights=(0.25, 0.25, 0.25, 0.25),smoothfunc=item)

        bleu1(ref_list, hyp_list)

        #bleu2(ref_list, hyp_list)
        bleu4(ref_list, hyp_list)
        bleu_aver(ref_list, hyp_list)

        #print(round(meteor_score(['你 好 啊'], '你 和 我 都 好', 4)))

        print("BLEU1", bleu1.get_metric(reset=False))
        print("BLEU4", bleu4.get_metric(reset=False))
        print("BLEU_avg", bleu_aver.get_metric(reset=False))



        # bleu_aver(ref_list, hyp_list)
    #print("BLEU1", bleu1.get_metric(reset=False))
    # print("BLEU2", bleu2.get_metric(reset=False))
    #print("BLEU4", bleu4.get_metric(reset=False))
    #print("BLEU_avg", bleu_aver.get_metric(reset=False))
    # print(compute_bleu(open('ref.txt','r',encoding='utf-8').read(),open('ref.txt','r',encoding='utf-8').read()))
    print(metric)
    # print(len(hyp_list),len(ref_list))


def main():
    if args.intent_evaluation:
        intent_evaluation()
    if args.action_evaluation:
        action_evaluation()
    if args.generate_evaluation:
        #generate_gpt()
        #intent_evaluation()
        action_evaluation()
        generate_text()


if __name__ == '__main__':
    main()
