
import argparse
import os
import torch
import logging
from datetime import datetime
#import ipdb


class ArgsParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser = argparse.ArgumentParser()
        parser.add_argument('--device', default='0,1', type=str, required=False, help='设置使用哪些显卡')
        parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
        parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str,
                            required=False,
                            help='选择模型参数')
        parser.add_argument('--vocab_path', default='vocabulary/vocab_small.txt', type=str, required=False, help='选择词库')
        parser.add_argument('--train_raw_path', default='data/train.txt', type=str, required=False, help='原始训练语料')
        parser.add_argument('--train_tokenized_path', default='data/train_tokenized.txt', type=str,
                            required=False,
                            help='将原始训练语料tokenize之后的数据的存放位置')
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
        parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False,
                            help='Tensorboard路径')
        parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
        parser.add_argument('--num_workers', type=int, default=1, help="dataloader加载数据时使用的线程数量")
        parser.add_argument('--train_mmi', action='store_true', help="若指定该参数，则训练DialoGPT的MMI模型")
        parser.add_argument('--train_mmi_tokenized_path', default='data/train_mmi_tokenized.txt', type=str,
                            required=False,
                            help='将原始训练语料的每段对话翻转，然后进行tokenize之后的数据的存放位置，用于训练MMI模型')
        parser.add_argument('--mmi_model_output_path', default='mmi_model', type=str, required=False, help='MMI模型保存路径')
        parser.add_argument('--evaluation_path', default='result.json', type=str, required=False, help='评测时文件路径')
        parser.add_argument('--checkpoint', default='dialogue_small_dialogue/model_epoch18', type=str, required=False, help='生成时模型路径')
        parser.add_argument('--eval_all_checkpoints', action='store_true',help='在所有模型上评价')
        parser.add_argument('--test_path', default='data/test_generation.json', help='测试文件路径')
        parser.add_argument('--generate', action='store_true', help='分别生成回复的文件和参考文件')
        parser.add_argument('--generate_evaluation', action='store_true', help='评测生成指标')
        parser.add_argument('--intent_evaluation', action='store_true', help='评测intent指标')
        parser.add_argument('--action_evaluation', action='store_true', help='评测action指标')
        parser.add_argument('--input_type', default='predicted', help='生成测试集的选项')
        # parser.add_argument('--max_len', type=int, default=60, help='每个utterance的最大长度,超过指定长度则进行截断')
        # parser.add_argument('--max_history_len', type=int, default=4, help="dialogue history的最大长度")
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()

        return args

