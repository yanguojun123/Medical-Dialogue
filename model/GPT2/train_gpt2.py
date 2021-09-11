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
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.utils.data.distributed import DistributedSampler
import sys,re
#from models.gpt2 import GPT2LMHeadModel
from models.gpt2 import GPT2Config
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import shutil

PAD = '[PAD]'
pad_id = 0
logger = None

def repeatAction(str):
    """
    Judge whether duplicate action slot values are generated
    :param str: Generated string
    :return:Return True or False
    """
    if len(set(str.split('<|continue|>')))<len(str.split('<|continue|>')):
        return True
    else:
        return False

def setup_train_args():
    """
    Set training parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1', type=str, required=False, help='Set which graphics cards to use')
    parser.add_argument('--no_cuda', action='store_true', help='Training without GPU')
    parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
                        help='Select model config file')
    parser.add_argument('--vocab_path', default='vocabulary/vocab_small.txt', type=str, required=False, help='Select vocabulary file')
    parser.add_argument('--train_raw_path', default='data/train.txt', type=str, required=False, help='Original training corpus')
    parser.add_argument('--dev_raw_path', default='data/dev.txt', type=str, required=False, help='Original dev corpus')
    parser.add_argument('--train_tokenized_path', default='data/train_tokenized.txt', type=str,
                        required=False,
                        help='The storage location of the data after tokenizing the original training corpus')
    parser.add_argument('--dev_tokenized_path', default='data/dev_tokenized.txt', type=str,
                       required=False,
                       help='The storage location of the data after tokenizing the original dev corpus')
    parser.add_argument('--log_path', default='data/training.log', type=str, required=False, help='Training log storage location')
    parser.add_argument('--raw', action='store_true', help='Tokenize the original training corpus')
    parser.add_argument('--epochs', default=10, type=int, required=False, help='epoch numbers')
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
    parser.add_argument("--local_rank", type=int, default=-1, help='distributed')
    parser.add_argument("--continue_epoch", type=int, default=0, help='continue train')
    parser.add_argument("--change_parameter", action='store_true', help='change pretrained model parameter')
    parser.add_argument("--ft2", action='store_true', help='second fine tune or third fine-tune')
    parser.add_argument("--inference_type", type=str, default='groundtruth', required=False, help='generate type: predicted or groundtruth')
    parser.add_argument("--inference_result", type=str, default='../data/result_file.json', required=False,
                        help='file to save inference result')
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


def create_model(args, vocab_size):
    """
    Get model and maximum truncation length
    :param args:
    :param vocab_size
    :return:model and
    """
    if args.pretrained_model:  # If a pre trained  model is specified

        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
        '''pe = model.state_dict['transformer.wpe.weight']
        ab = model.state_dict['transformer.h.0.attn.bias']
        print("pe:", pe[0:50, :])
        print("ab", ab[0:50, :])
        exit(-1)'''
        if args.change_parameter:# Change the maximum truncation length of the loaded pre training model
            '''model_config_temp = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
            model_temp = GPT2LMHeadModel(config=model_config_temp)
            pe = model_temp.state_dict()['transformer.wpe.weight']'''
            pe = model.state_dict()['transformer.wpe.weight']
            print("pe1:", pe[:50, :])
            pe = torch.zeros(800, 768)
            #pe_lm = torch.nn.Linear(300, 800)
            #pe = pe_lm(model.state_dict()['transformer.wpe.weight'].transpose(0, 1)).transpose(0, 1)
            torch.nn.init.xavier_uniform_(pe)
            model.resize_token_embeddings(vocab_size)
            pretrained_dict = {"transformer.wpe.weight": pe}
            print("pe2:", pe[:50, :])
            model_dict = model.state_dict()
            for id in range(10):
                pretrained_dict['transformer.h.'+str(id)+'.attn.bias'] = torch.tril(torch.ones(800, 800)).view(1, 1, 800, 800)
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            #print(pretrained_dict)
            model_dict.update(pretrained_dict)
            #model.save_pretrained()
            for k, v in model_dict.items():
                print(k, v.size())
            # 3. load the new state dict
            print('config:', json.load(open(args.model_config, 'r')))
            model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path=None, config=args.model_config, state_dict=model_dict)

    else:  # If no pre training model is specified, the model is initialized
        model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
        model = GPT2LMHeadModel(config=model_config)
        model_dict = model.state_dict()
        pe = model_dict['transformer.wpe.weight']
        ab = model_dict['transformer.h.0.attn.bias']
        print("pe:", pe[0:50, :])
        print("ab", ab[0:50, :])
        exit(-1)
    # Adjust the size of voca of gpt2 model according to the vocabulary of tokenizer
    model.resize_token_embeddings(vocab_size)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    return model, model.config.to_dict().get("n_ctx")


def preprocess_raw_data(raw_path, tokenized_path, tokenizer, n_ctx):
    """
    Process the original corpus and convert the original corpus into token ID for train
    :param raw_path: Original corpus
    :param tokenizer_path:Token id file path corresponding to the original corpus
    :param tokenizer:Tokenizer object of pre training model
    :param n_ctx:Maxmium truncation lenght
    :return:
    """
    logger.info("tokenizing raw data,raw data path:{}, token output path:{}".format(raw_path,
                                                                                    tokenized_path))
    special_token = tokenizer.encode(
        '<|intent|> <|endofintent|> <|action|> <|endofaction|> <|response|> <|endofresponse|>')[1:-1]
    special_token[0] = tokenizer.encode('<|intent|>')[1]
    special_token[1] = tokenizer.encode('<|endofintent|>')[1]
    special_token[2] = tokenizer.encode('<|action|>')[1]
    special_token[3] = tokenizer.encode('<|endofaction|>')[1]
    special_token[4] = tokenizer.encode('<|response|>')[1]
    special_token[5] = tokenizer.encode('<|endofresponse|>')[1]
    user_id = tokenizer.encode('<|user|>')[1]
    currentuser_id = tokenizer.encode('<|currentuser|>')[1]
    miss_count = 0
    with open(raw_path, 'rb') as f:
        data = f.read().decode("utf-8")
    if "\r\n" in data:
        train_data = data.split("\r\n\r\n")
    else:
        train_data = data.split("\n\n")
    logger.info("there are {} dialogue in raw dataset".format(len(train_data)))
    with open(tokenized_path, "w", encoding="utf-8") as f:
        for dialogue_index, dialogue in enumerate(tqdm(train_data)):
            if "\r\n" in data:
                utterances = dialogue.split("\r\n")
            else:
                utterances = dialogue.split("\n")

            for utterance in utterances:
                try :
                    if utterance !='':
                        utterance = '<|endoftext|> '+ utterance.split('<|endofcontext|>')[1]
                    special_index = [0] * 6
                    dialogue_ids = tokenizer.encode(utterance)[1:-1]


                    user_count = 0
                    if n_ctx-6 < len(dialogue_ids):#
                        for index, id in enumerate(dialogue_ids):
                            if id == currentuser_id:
                                currentuser_index = index
                            if id == user_id and user_count == 2:
                                user_index = index
                                #dialogue_ids = dialogue_ids[0:index]+dialogue_ids[-(n_ctx-6-index):]
                            if id == user_id and user_count < 2:
                                user_count += 1

                        if len(dialogue_ids[currentuser_index:]) > n_ctx-6:
                            miss_count += 1
                            continue
                        elif len(dialogue_ids[0:user_index])+len(dialogue_ids[currentuser_index:]) > n_ctx-6 :
                             dialogue_ids = tokenizer.encode('<|endoftext|> <|context|> <|endofcontext|>')+ dialogue_ids[currentuser_index:]
                        else:
                            dialogue_ids = dialogue_ids[0:user_index] + dialogue_ids[-(n_ctx - 6 - index):]
                    # print(len(dialogue_ids))
                    # dialogue_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in utterance])
                    # dialogue_ids.extend(tokenizer.encode(utterance))
                    # dialogue_ids.append(tokenizer.sep_token_id)
                    # print(len(dialogue_ids))
                    #dialogue_ids = dialogue_ids[-n_ctx:]
                    # Perform subscript search for three special flags
                    for index, id in enumerate(dialogue_ids):
                        if id == special_token[0]:
                            special_index[0] = index
                        if id == special_token[1]:
                            special_index[1] = index
                        if id == special_token[2]:
                            special_index[2] = index
                        if id == special_token[3]:
                            special_index[3] = index
                            special_index[4] = index + 1
                    special_index[5] = len(dialogue_ids) - 2
                    if special_index[0] == 0:
                        continue
                    special_index.extend(dialogue_ids)  # Add the subscripts of 6 special tokens to the front

                    for id in special_index:
                        f.write(str(id) + ' ')
                    #The last record does not add a newline character
                    if dialogue_index < len(train_data) - 1:
                        f.write("\n")
                except IndexError:
                    pass
        print("miss_count:", miss_count)
    logger.info("finish preprocessing raw data,the result is stored in {}".format(tokenized_path))

def calculate_loss_and_accuracy(outputs, labels, device, special_index):
    """
    Calculate non pad_id average loss and accuracy
    :param outputs:Model preiction score
    :param labels:Label of current data
    :param device: CPU or GPU
    :return: Loss and accuracy
    """

    # tb_writer = SummaryWriter(log_dir=args.writer_dir)
    logits = outputs[0]  # Prediction score of the next token,size:[batch_size,token_len,voca_size]
    # Use the first n-1 tokens to predict the nth token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    # special_index=special_index-np.ones((len(special_index),6),dtype=int)
    loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')  #
    # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),shift_labels.view(-1))

    intent_index = torch.zeros(labels.size(0), labels.size(1) - 1)
    action_index = torch.zeros(labels.size(0), labels.size(1) - 1)
    response_index = torch.zeros(labels.size(0), labels.size(1) - 1)
    intent_num = 0
    action_num = 0
    response_num = 0
    for count in range(len(special_index)):  # Calculate the three output ranges of each batch separately
        intent_temp = torch.zeros(labels.size(-1) - 1)
        action_temp = torch.zeros(labels.size(-1) - 1)
        response_temp = torch.zeros(labels.size(-1) - 1)
        '''
        intent_begin_index=0
        intent_end_index = 0
        action_begin_index = 0
        action_end_index = 0
        response_begin_index = 0
        response_end_index = 0

        for count enumerate(labels[count]):

            y=y.item()
            if y==tokenizer.encode('<|intent|>')[0]:
                intent_begin_index=index+1
            if y==tokenizer.encode('<|endofintent|>')[0]:
                intent_end_index=index-1
            if y==tokenizer.encode('<|action|>')[0]:
                action_begin_index=index+1
            if y==tokenizer.encode('<|endofaction|>')[0]:
                action_end_index=index-1
            if y==tokenizer.encode('<|response|>')[0]:
                response_begin_index=index+1
            if y==tokenizer.encode('<|endofresponse|>')[0]:
                response_end_index=index-1
        if intent_begin_index>intent_end_index:        
           intent_end_index=labels.size(1)-1 
        if  action_begin_index>action_end_index: 
           action_end_index=labels.size(1)-1 
        if  response_begin_index>response_end_index: 
           response_end_index=labels.size(1)-1 

        action_num+=action_end_index - action_begin_index + 1
        intent_num+=intent_end_index-intent_begin_index+1
        response_num+=response_end_index - response_begin_index + 1     
        intent_temp[intent_begin_index:min(intent_end_index+1,labels.size(1)-1)]=torch.ones((min(intent_end_index,labels.size(1)-2)-intent_begin_index+1))
        action_temp[action_begin_index:min(action_end_index+1,labels.size(1)-1)] =torch.ones((min(action_end_index,labels.size(1)-2) - action_begin_index + 1))
        response_temp[response_begin_index:min(response_end_index+1,labels.size(1)-1)] = torch.ones((min(response_end_index,labels.size(1)-2) - response_begin_index + 1))
        '''
        intent_num += special_index[count][1] - special_index[count][0] + 1
        action_num += special_index[count][3] - special_index[count][2] + 1
        response_num += special_index[count][5] - special_index[count][4] + 1

        intent_temp[special_index[count][0]:special_index[count][1] + 1] = torch.ones(
            special_index[count][1] - special_index[count][0] + 1)
        action_temp[special_index[count][2]:special_index[count][3] + 1] = torch.ones(
            special_index[count][3] - special_index[count][2] + 1)
        response_temp[special_index[count][4]:special_index[count][5] + 1] = torch.ones(
            special_index[count][5] - special_index[count][4] + 1)
        intent_index[count] = intent_temp
        action_index[count] = action_temp
        response_index[count] = response_temp

        # print('step:',count)
    intent_index = intent_index.long().contiguous().to(device)
    action_index = action_index.long().contiguous().to(device)
    response_index = response_index.long().contiguous().to(device)
    intent_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                           (shift_labels * intent_index).view(-1)) / intent_num
    action_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                           (shift_labels * action_index).view(-1)) / action_num
    response_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                             (shift_labels * response_index).view(-1)) / response_num
    _, preds = shift_logits.max(dim=-1)  # size:[batch_size,token_len]

    # tb_writer.add_scalar('intent_loss',intent_loss.item(),)
    # print('intent_loss:',intent_loss.item(),'action_loss',action_loss.item(),'response_loss',response_loss.item())
    not_ignore = shift_labels.ne(pad_id)  # Perform non operation and return a tensor if the i-th position of view is
    # pad_id, it is set to 0, otherwise it is 1
    num_targets = not_ignore.long().sum().item()

    correct = (shift_labels == preds) & not_ignore  # Calculate the number of correct tokens predicted by the model and exclude the tokens of the pad
    correct = correct.float().sum()

    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))
    accuracy = correct / (num_targets)
    #loss = loss / num_targets
    #loss = intent_loss
    #loss = action_loss
    #loss = response_loss
    #loss = intent_loss + action_loss + response_loss
    loss = [loss, intent_loss, action_loss, response_loss]
    return loss, accuracy


def collate_fn(batch):
    """
    Calculate the longest input of all samples in the batch, and align the length of other inputs to it
    :param batch:Batch data fetched by dataloder each time
    :return:Data in tensor format
    """
    global pad_id
    input_ids = []
    btc_size = len(batch)
    # if btc_size!=32:
    #    print(batch)
    max_input_len = 0  # The longest input in the batch is used for data alignment of the batch
    # Calculate the maximum length of input in the batch
    for btc_idx in range(btc_size):
        if max_input_len < len(batch[btc_idx]):
            max_input_len = len(batch[btc_idx])
    # Using pad_id to complete the length less than max_input_id
    for btc_idx in range(btc_size):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend([pad_id] * (max_input_len - input_len))
    return torch.tensor(input_ids, dtype=torch.long)


def train(model, device, train_list, multi_gpu, args, tokenizer, tb_writer,overall_step):
    """
    Train the model
    :param model:Pretrained model
    :param device:CPU or GPU
    :param train_list:train set
    :param multi_gpu:Is it multi GPU training
    :param args:Experimental parameters
    :param tokenizer:Tokenizer object of pre training model
    :param tb_writer:Tensorboard writer file object
    :param overall_step:Total training steps
    :return:NULL
    """

    train_dataset = MyDataset(train_list)
    overall_step = 0
    '''# multi-gpu training
    if multi_gpu:
        model = torch.nn.DataParallel(model)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )'''

    model.train()

    # Calculate the total number of steps for parameter optimization of all epochs steps
    total_steps = int(train_dataset.__len__() * args.epochs / args.batch_size / args.gradient_accumulation)
    logger.info('total training steps = {}'.format(total_steps))

    #Set up the optimizer and use the warmup policy at the initial training
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,num_training_steps=total_steps)

    logger.info('starting training')
    # Used to count the accumulated loss of each gradient
    running_loss = 0


    # Number of times to record out of memory
    oom_time = 0
    # 开始训练
    for epoch in range(args.continue_epoch,args.epochs):
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
            #print(input_ids)
            special_index = (input_ids[:, 0:6] - np.ones((input_ids.size(0), 6), dtype=int)).to(device)
            # special_index = input_ids[:, 0:6].to(device)
            input_ids = input_ids[:, 6:].to(device)

            '''
            #attention_mask, Change the mask matrix of the model
            attention_mask = torch.ones()
            temp_count =0
            mask_matrix = torch.ones(input_ids.size(0), 1,1,input_ids.size(1), input_ids.size(1))
            for x in special_index:
                #print(x[0])
                temp = x[0]
                if temp < 1:
                    temp = 10
                mask_1 = torch.ones(input_ids.size(-1),temp) #left
                mask_2 = torch.zeros(temp,input_ids.size(-1)-temp) # right_upper
                mask_3 = torch.tril(torch.ones(input_ids.size(-1)-temp, input_ids.size(-1)-temp))
                mask_right = torch.cat((mask_2, mask_3),0)
                mask_all = torch.cat((mask_1, mask_right),1)
                mask_matrix[temp_count][0][0] =mask_all
                temp_count += 1
            print("attention_mask:", attention_mask.size(), attention_mask)'''

            #Solve the CUDA out of memory problem caused by insufficient video memory during operation
            try:
                outputs = model.forward(input_ids=input_ids)

                lossall, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids, device=device,
                                                                tokenizer=tokenizer, args=args,
                                                                special_index=special_index)
                loss = lossall[0]
                intent_loss = lossall[1]
                action_loss = lossall[2]
                response_loss = lossall[3]
                if multi_gpu:
                    loss = loss.mean()
                    intent_loss = intent_loss.mean()
                    action_loss = action_loss.mean()
                    response_loss = response_loss.mean()
                    accuracy = accuracy.mean()
                if args.gradient_accumulation > 1:
                    loss = loss / args.gradient_accumulation
                    intent_loss = intent_loss / args.gradient_accumulation
                    action_loss = action_loss / args.gradient_accumulation
                    response_loss = response_loss / args.gradient_accumulation
                    accuracy = accuracy / args.gradient_accumulation
                loss.backward()
                # intent_loss.backward()
                # action_loss.backward()
                # response_loss.backward()
                # Gradient clipping solves the problem of gradient disappearance or explosion, that is, setting the threshold
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # After the gradient accumulation of a certain step, update the parameters
                if (batch_idx + 1) % args.gradient_accumulation == 0:
                    running_loss += loss.item()
                    # Update parameters
                    optimizer.step()
                    # Clear gradient information
                    optimizer.zero_grad()
                    # Perform warm up
                    scheduler.step()
                    overall_step += 1
                    # Update log and tnesorboardx information
                    if (overall_step + 1) % args.log_step == 0 and args.local_rank == 0:
                        logger.info(
                            "batch {} of epoch {}, loss {}, intent_loss {}, action_loss {}, response_loss {}, accuracy {}".format(
                                batch_idx + 1, epoch + 1, loss,
                                intent_loss, action_loss, response_loss, accuracy))
                        tb_writer.add_scalar('loss', loss.item(), overall_step)
                        tb_writer.add_scalar('intent_loss', intent_loss.item(), overall_step)
                        tb_writer.add_scalar('action_loss', action_loss.item(), overall_step)
                        tb_writer.add_scalar('response_loss', response_loss.item(), overall_step)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    oom_time += 1
                    logger.info("WARNING: ran out of memory,times: {}".format(oom_time))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    logger.info(str(exception))
                    raise exception
        logger.info('saving model for epoch {}'.format(epoch + 1))
        if args.local_rank == 0:
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


def evaluate(model, device, dev_list, multi_gpu, args, tokenizer, tb_writer, overall_step):
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
    overall_step += 10


    loss_epoch = [0] * 4
    accuracy_epoch = 0
    # tb_writer = SummaryWriter(log_dir=args.writer_dir)
    test_dataset = MyDataset(dev_list)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                 collate_fn=collate_fn)
    with torch.no_grad():

        for batch_idx, input_ids in enumerate(test_dataloader):
            special_index = (input_ids[:, 0:6] - np.ones((input_ids.size(0), 6), dtype=int)).to(device)
            if args.cls_token:
                cls_token = tokenizer.cls_token_id * torch.ones(input_ids.size(0), dtype=torch.long).unsqueeze(1)
                input_ids = torch.cat((cls_token, input_ids[:, 6:]), 1).to(device)
            else:
                input_ids = input_ids[:, 6:].to(device)
            # input_ids.to(device)

            outputs = model.forward(input_ids=input_ids)
            lossall, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids, device=device,
                                                            special_index=special_index, tokenizer=tokenizer, args=args)

            loss = lossall[0]
            loss_epoch[0] += loss
            intent_loss = lossall[1]
            action_loss = lossall[2]
            response_loss = lossall[3]
            loss_epoch[1] += intent_loss
            loss_epoch[2] += action_loss
            loss_epoch[3] += response_loss
            overall_step += 1
            accuracy_epoch += accuracy
            if multi_gpu:
                loss = loss.mean()
                intent_loss = intent_loss.mean()
                action_loss = action_loss.mean()
                response_loss = response_loss.mean()
                accuracy = accuracy.mean()
            if args.gradient_accumulation > 1:
                loss = loss / args.gradient_accumulation
                intent_loss = intent_loss / args.gradient_accumulation
                action_loss = action_loss / args.gradient_accumulation
                response_loss = response_loss / args.gradient_accumulation
                accuracy = accuracy / args.gradient_accumulation
            if (batch_idx % args.log_step) == 0 and args.local_rank == 0:
                logger.info(
                    "evaluate batch {} ,loss {} ,intent_loss {}, action_loss {},response_loss {},accuracy {}".format(
                        batch_idx, loss, intent_loss, action_loss, response_loss, accuracy))
                tb_writer.add_scalar('loss', loss.item(), overall_step)
                tb_writer.add_scalar('intent_loss', intent_loss.item(), overall_step)
                tb_writer.add_scalar('action_loss', action_loss.item(), overall_step)
                tb_writer.add_scalar('response_loss', response_loss.item(), overall_step)
        batch_num = len(test_dataloader)
        logger.info("finishing evaluating. loss {},intent_loss {},action_loss {},response_loss {}, accuracy {}".format(
            loss_epoch[0] / batch_num, loss_epoch[1] / batch_num, loss_epoch[2] / batch_num, loss_epoch[3] / batch_num,
            accuracy_epoch / batch_num))

    return loss_epoch[0]


def generate(tokenizer, model, args):
    """
    To use trained model to generate result
    :param tokenizer:Tokenizer
    :param model:Model to be used
    :param args:Parameters list
    :return:
    """
    model = GPT2LMHeadModel.from_pretrained(model)
    model.eval()
    model.to('cuda')
    break_tokens = tokenizer.encode('<|endoftext|>')[1:-1]
    MAX_LEN = model.config.n_ctx

    datas = open(args.test_path, 'r', encoding='utf-8').read().split('\n\n')
    generated_dict = {}
    num_data = len(datas)
    for i, data in enumerate(datas[0:-1]):

        print('[{}/{}] \r'.format(i, num_data), end='')
        sys.stdout.flush()
        #  system_delex = d_delex['target_raw']

        dialogue_target_intent = []
        dialogue_target_action = []
        dialogue_target_response = []
        generated = []
        generated_intents = []
        generated_actions = []
        generated_responses = []
        for turn_id, turn in enumerate(data.split('\n')):

            text = turn.split('<|intent|>')[0]
            target_intent = turn.split('<|intent|>')[1].split('<|endofintent|>')[0]
            target_action = turn.split('<|action|>')[1].split('<|endofaction|>')[0]
            target_response = turn.split('<|response|>')[1].split('<|endofresponse|>')[0]
            knowledge = turn.split('<|endofintent|>')[1].split('<|action|>')[0]

            dialogue_target_intent.append(target_intent)
            dialogue_target_action.append(target_action)
            dialogue_target_response.append(target_response)
            # model_context.append(text)
            indexed_tokens = tokenizer.encode(text)[1:-1]
            if len(indexed_tokens) > MAX_LEN:
                indexed_tokens = indexed_tokens[-1 * MAX_LEN:]

            # Convert indexed tokens in a PyTorch tensor
            tokens_tensor = torch.tensor([indexed_tokens])

            # If you have a GPU, put everything on cuda
            tokens_tensor = tokens_tensor.to('cuda')
            predicted_index = indexed_tokens[-1]
            with torch.no_grad():
                if args.inference_type == 'predicted':
                    # truncate_action = False
                    get_intent = False
                    get_action = False
                    while predicted_index not in break_tokens:
                        outputs = model(tokens_tensor)
                        predictions = outputs[0]
                        predicted_index = torch.argmax(predictions[0, -1, :]).item()
                        indexed_tokens += [predicted_index]

                        # sometime model generate repeated actions, we just use truncate actions if this happens
                        predicted_text = tokenizer.decode(indexed_tokens)
                        if '<|intent|>' in predicted_text and (not get_intent) and repeatAction(
                                predicted_text.split('<|intent|>')[1]):
                            get_intent = True
                            # generated_intents.append('<|continue|>'.join(predicted_text.split('<|intent|>')[1].split('<|continue|>')[0:-2]))
                            indexed_tokens = tokenizer.encode(
                                predicted_text.split('<|intent|>')[0] + '<|intent|> {} <|endofintent|>'.format(
                                    '<|continue|>'.join(predicted_text.split('<|intent|>')[1].split('<|continue|>')[
                                                        0:-2])) + knowledge)[1:-1]

                        if '<|action|>' in predicted_text and (not get_action) and repeatAction(
                                predicted_text.split('<|action|>')[1]):
                            get_action = True
                            # generated_actions.append(
                            #   '<|continue|>'.join(predicted_text.split('<|action|>')[1].split('<|continue|>')[:-2]))
                            indexed_tokens = tokenizer.encode(
                                predicted_text.split('<|action|>')[0] + '<|action|> {} <|endofaction|>'.format(
                                    '<|continue|>'.join(
                                        predicted_text.split('<|action|>')[1].split('<|continue|>')[:-2])))[1:-1]
                        '''if '<|action|>' in predicted_text:
                            generated_actions = predicted_text.split('<|action|>')[-1].split('<|endofaction|>')[
                                0].split('<|continue|>')
                            new_actions = []
                            for a in generated_actions:
                                if a in ['', ' ']:
                                    continue
                                new_actions.append(a.strip())
                            len_actions = len(new_actions)
                            if len(list(set(new_actions))) > len(new_actions) or (
                                    len_actions > 10 and not truncate_action):
                                actions = '<|action|> {} <|endofaction|>'.format(' , '.join(list(set(new_actions))))
                                indexed_tokens = tokenizer.encode(
                                    '{} {}'.format(predicted_text.split('<|action|>')[0], actions))[1:-1]
                                truncate_action = True'''
                        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
                        if len(indexed_tokens) > MAX_LEN:
                            break
                        if tokenizer.decode(indexed_tokens).endswith('<|endofresponse|>'):
                            break

                    predicted_text = tokenizer.decode(indexed_tokens)
                    # tmp = ' '.join([predicted_text.split('<|endofresponse|>')[0], '<|endofresponse|>'])
                    generated.append(predicted_text)
                # Generate a new section with the ground truth as input
                if args.inference_type == 'groundtruth':
                    get_intent = False
                    get_action = False
                    predicted_text = ''
                    while predicted_index not in break_tokens:
                        outputs = model(tokens_tensor)
                        predictions = outputs[0]
                        predicted_index = torch.argmax(predictions[0, -1, :]).item()
                        indexed_tokens += [predicted_index]

                        temp = re.sub(' ', '', tokenizer.decode(predicted_index))
                        print("temp:", temp)
                        '''if temp != '':
                            if  get_action and temp in predicted_text.split('<|endofaction|>')[1]:
                                print("predicted_text:",predicted_text)
                                indexed_tokens += tokenizer.encode('<|endofresponse|>')[1:-1]
                            else:
                                indexed_tokens += [predicted_index]
                        # print(predicted_index, tokenizer.decode(predicted_index))
                        else:
                            indexed_tokens += [predicted_index]'''

                        # sometime model generate repeated actions, we just use truncate actions if this happens
                        predicted_text = tokenizer.decode(indexed_tokens)
                        # Remove the generated intent, and then add the ground truth
                        if '<|intent|>' in predicted_text and (not get_intent) and repeatAction(
                                predicted_text.split('<|intent|>')[1]):
                            get_intent = True
                            generated_intents.append('<|continue|>'.join(
                                predicted_text.split('<|intent|>')[1].split('<|continue|>')[0:-2]))
                            indexed_tokens = tokenizer.encode(
                                predicted_text.split('<|intent|>')[0] + '<|intent|> {} <|endofintent|>'.format(
                                    target_intent) + knowledge)[1:-1]

                        if '<|endofintent|>' in predicted_text and not get_intent:
                            if '<|intent|>' in predicted_text:
                                generated_intents.append(
                                    predicted_text.split('<|intent|>')[1].split('<|endofintent|>')[0])
                                get_intent = True
                                indexed_tokens = tokenizer.encode(
                                    predicted_text.split('<|intent|>')[0] + '<|intent|> {} <|endofintent|>'.format(
                                        target_intent) + knowledge)[1:-1]

                        if '<|action|>' in predicted_text and (not get_action) and repeatAction(
                                predicted_text.split('<|action|>')[1]):
                            get_action = True
                            generated_actions.append('<|continue|>'.join(
                                predicted_text.split('<|action|>')[1].split('<|continue|>')[:-2]))
                            indexed_tokens = tokenizer.encode(
                                predicted_text.split('<|action|>')[0] + '<|action|> {} <|endofaction|>'.format(
                                    target_action))[1:-1]

                        if '<|endofaction|>' in predicted_text and not get_action:  # 去除生成的action，再加入groundtruth
                            if '<|action|>' in predicted_text:
                                generated_actions.append(
                                    predicted_text.split('<|action|>')[1].split('<|endofaction|>')[0])
                                get_action = True
                                indexed_tokens = tokenizer.encode(
                                    predicted_text.split('<|action|>')[0] + '<|action|> {} <|endofaction|>'.format(
                                        target_action))[1:-1]
                        '''if '<|endofaction|>' in predicted_text and not get_action:
                            get_action = True
                            generated_actions.append(
                                predicted_text.split('<|action|>')[1].split('<|endofaction|>')[0])
                            indexed_tokens = tokenizer.encode(
                                predicted_text.split('<|action|>')[0] + '<|action|> {} <|endofaction|> <|response|>'.format(
                                    target_action))[1:-1]'''
                        predicted_text = re.sub(' ', '', tokenizer.decode(indexed_tokens))
                        # print("predicted_text:",predicted_text)

                        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
                        if len(indexed_tokens) > MAX_LEN:
                            break
                        if predicted_text.endswith('<|endofresponse|>'):
                            break
                    if not get_intent:
                        generated_intents.append(' ')
                    if not get_action:
                        generated_actions.append(' ')
                    predicted_text = re.sub(' ', '', tokenizer.decode(indexed_tokens))
                    print("predicted_text:", predicted_text)
                    if '<|response|>' in predicted_text and '<|endofresponse|>' in predicted_text:
                        generated_responses.append(
                            predicted_text.split('<|response|>')[1].split('<|endofresponse|>')[0])
                    else:
                        generated_responses.append(' ')

        if args.input_type == 'predicted':
            dialogue_pred_intent = []
            dialogue_pred_responses = []
            dialogue_pred_action = []
            for x in generated:
                if '<|intent|>' in x and '<|endofintent|>' in x:
                    dialogue_pred_intent.append(x.split('<|intent|>')[1].split('<|endofintent|>')[0])
                else:
                    dialogue_pred_intent.append(' ')
                if '<|action|>' in x and '<|endofaction|>' in x:
                    dialogue_pred_action.append(x.split('<|action|>')[1].split('<|endofaction|>')[0])
                else:
                    dialogue_pred_action.append(' ')
                if '<|response|>' in x and '<|endofresponse|>' in x:
                    dialogue_pred_responses.append(x.split('<|response|>')[1].split('<|endofresponse|>')[0])
                else:
                    dialogue_pred_responses.append(' ')
        else:
            dialogue_pred_intent = generated_intents
            dialogue_pred_action = generated_actions
            dialogue_pred_responses = generated_responses
        generated_dict[i] = {
            'target_intent': dialogue_target_intent,
            'generated_intent': dialogue_pred_intent,
            'target_action': dialogue_target_action,
            'generated_action': dialogue_pred_action,
            'target_response': dialogue_target_response,
            'generated_response': dialogue_pred_responses
        }
    with open('{}.json'.format(args.inference_result), 'wt', encoding='utf-8') as f:
        json.dump(generated_dict, f, ensure_ascii=False)
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

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.seed:
        set_random_seed(args)
    # Log output to both file and console
    global logger
    logger = create_logger(args)
    multi_gpu = ''
    tb_writer = ''
    logger.info(args)
    tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    tokenizer.add_special_tokens({'bos_token': '<|endoftext|>'})
    tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<|context|>', '<|endofcontext|>', '<|user|>', '<|system|>',
                                       '<|currentuser|>',
                                       '<|continue|>', '<|intent|>', '<|endofintent|>', '<|action|>',
                                       '<|endofcurrentuser|>',
                                       '<|endofaction|>', '<|response|>', '<|endofresponse|>','<|knowledge|>', '<|endofknowledge|>', '<|k|>', 'Inform',
                                       'Inquire', 'Recommend', 'Diagnosis', 'Chitchat', 'Other', 'disease',
                                       'symptom',
                                       'treatment', 'other', 'department'
            , 'time', 'precaution', 'QuestionAnswering', 'medicine', 'pathogeny', 'side_effect', 'effect',
                                       'temperature', 'range_body'
            , 'degree', 'frequency', 'dose', 'check_item', 'medicine_category', 'medical_place',
                                       'disease_history']})
    # vocabulary size of tokenizer
    vocab_size = len(tokenizer)
    #vocab_size = 13315
    global pad_id
    pad_id = tokenizer.convert_tokens_to_ids(PAD)
    overall_step = 0
    # Setup CUDA, GPU & distributed training
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    #model = torch.nn.Linear(10,10).to(device)
    model, n_ctx = create_model(args, vocab_size)
    model.to(device)
    #if args.local_rank not in [-1, 0]:
    #   torch.distributed.barrier()  # if not the first process, do not load pretrained model & vocab
    if args.local_rank == 0:

        # Record the number of model parameters
        num_parameters = 0
        parameters = model.parameters()
        for parameter in parameters:
            num_parameters += parameter.numel()
        logger.info('number of model parameters: {}'.format(num_parameters))
        # Preprocess the original data and convert the original corpus into the corresponding token_id
        if args.raw:
            preprocess_raw_data(args.train_raw_path, args.train_tokenied_path, tokenizer, n_ctx)
            preprocess_raw_data(args.dev_raw_path, args.dev_tokenied_path, tokenizer, n_ctx)
        tb_writer, multi_gpu= init(args)

    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    # load data
    logger.info("loading traing data")
    if args.raw:
        with open(args.train_tokenized_path, "r", encoding="utf8") as f:
            data_train = f.read()
        with open(args.dev_tokenized_path, "r", encoding="utf8") as d:
            data_dev = d.read()
    train_list = []
    test_list = []
    if args.ft2:
        for x in data_train.split("\n"):
            if len(x.split(' ')) <= 800:
                train_list.append(x)
        for x in data_dev.split("\n"):
            if len(x.split(' ')) <= 800:
                test_list.append(x)
    else:
        data_list = []
        for x in data_train.split('\n')[0:-1]:
            if len(x.split(' ')) < 800:
                data_list.append(x)
        train_list = data_list[int(0.13 * len(data_list)):]
        test_list = data_list[0:int(0.13 * len(data_list))]
    logger.info("Finish loading data")
    #if args.local_rank == 0:
    #    torch.distributed.barrier()  # finish barrier, when first process has loaded pretrained model & vocab

    '''# If GPU is available
    if torch.cuda.is_available():
        # Initialize distributed process group
        torch.distributed.init_process_group(backend='NCCL', init_method='env://')
        # If CUDA is available
    if torch.cuda.is_available() and args.local_rank is not None:
        print("GPU ", args.local_rank)
        # Build a distributed training model and use local Calculate the GPU of rank number and output the calculation results to local On GPU rank
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                                output_device=args.local_rank,
                                                                   find_unused_parameters=True)'''
    # start train

    train(model, device, train_list, multi_gpu, args, tokenizer, tb_writer, overall_step)
    # Model validation
    if args.local_rank == 0:
        best_model = ''
        if args.eval_all_checkpoints:
            checkpoints = [args.dialogue_model_output_path + c for c in
                           sorted(os.listdir(args.dialogue_model_output_path))[1:-1]]
            logger.info("Evaluate the following checkpoints: {}".format(checkpoints))
            #overstep = [0]
            min_loss = 100000
            for x in range(1, args.epochs + 1):
                checkpoint = args.dialogue_model_output_path + 'model_epoch' + str(x)
                model = GPT2LMHeadModel.from_pretrained(checkpoint)
                logger.info("Evaluate the checkpoint: {}".format(checkpoint))
                model.resize_token_embeddings(vocab_size)
                model.to(device)
                result = evaluate(model, device, test_list, multi_gpu, args, tokenizer, tb_writer, overall_step)
                if result < min_loss:
                    min_loss = result
                    best_model = checkpoint
            logger.info("the best model is " + best_model)
        tb_writer.close()
        generate(tokenizer, best_model, args)

if __name__ == '__main__':
    main()
