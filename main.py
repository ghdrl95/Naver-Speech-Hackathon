"""
Copyright 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#-*- coding: utf-8 -*-

import os
import sys
import time
import math
import wavio
import argparse
import queue
import shutil
import random
import math
import time
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import Levenshtein as Lev
from loader import _collate_fn
from evaluation.BeamSearch import testBeamSearch
from model import Jasper
import label_loader
from loader import *
from models import EncoderRNN, DecoderRNN, Seq2seq

import nsml
from nsml import GPU_NUM, DATASET_PATH, DATASET_NAME, HAS_DATASET
import toml


jasper_model_definition = toml.load('configs/jasper10x5dr_sp_offline_specaugment.toml')
char2index = dict()
index2char = dict()
SOS_token = 0
EOS_token = 0
PAD_token = 0

if HAS_DATASET == False:
    DATASET_PATH = './sample_dataset'

DATASET_PATH = os.path.join(DATASET_PATH, 'train')

def label_to_string(labels):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents

def char_distance(ref, hyp):
    ref = ref.replace(' ', '') 
    hyp = hyp.replace(' ', '') 

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length 

def get_distance(ref_labels, hyp_labels, display=False):
    total_dist = 0
    total_length = 0
    for i in range(len(ref_labels)):
        ref = label_to_string(ref_labels[i])
        hyp = label_to_string(hyp_labels[i])
        dist, length = char_distance(ref, hyp)
        total_dist += dist
        total_length += length 
        if display:
            cer = total_dist / total_length
            logger.debug('%d (%0.4f)\n(%s)\n(%s)' % (i, cer, ref, hyp))
    return total_dist, total_length

#반복글자 제거, blank라벨 제거
#minibatch만큼 반복
#blank,space가 아닌 첫번째 글자 추출 - 반복
#그 다음글자부터 같은 글자면 포함안시킴
#다른 글자면 저장한 글자 변경
#결과 리스트로 저장
def decoding_greedy(y_hat, blank=PAD_token, space = 662):
    all_sent = []
    max_t = y_hat.size(1)
    for data_setence in y_hat:
        current_word = -1
        #blank,space가 아닌 첫번째 글자 추출 - 반복
        for idx, data in enumerate(data_setence):
            if data.item() != blank and data.item() != space:
                current_word = data.item()
                break
        # 그 다음글자부터 같은 글자면 포함안시킴
        # 다른 글자면 저장한 글자 변경
        sent = []
        if current_word != -1:
            sent.append(current_word)
            for i in range(idx+1, len(data_setence)):
                data = data_setence[i].item()
                if data == blank or data == space or data == current_word:
                    continue
                current_word = data
                sent.append(current_word)
        #원래 칸수만큼 Blank 채우기
        for i in range(max_t - len(sent)):
            sent.append(space)
        all_sent.append(sent)
    return torch.LongTensor(all_sent)
def train(model, total_batch_size, train_datas, criterion, optimizer, device, train_begin, train_loader_count, print_batch=5, teacher_forcing_ratio=1):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    batch = 0

    model.train()

    logger.info('train() start')

    begin = epoch_begin = time.time()
    '''
    while True:
        if queue.empty():
            logger.debug('queue is empty')

        feats, scripts, feat_lengths, script_lengths = queue.get()

        if feats.shape[0] == 0:
            # empty feats means closing one loader
            train_loader_count -= 1

            logger.debug('left train_loader: %d' % (train_loader_count))

            if train_loader_count == 0:
                break
            else:
                continue
'''
    #logger.info("data size : %s)" %(len(train_datas)))
    for idx, data in enumerate(train_datas):

        feats, scripts, feat_lengths, script_lengths = data
        #logger.info("minibatch start : %s, data shape : %s frea_lengths : %s )" % (idx, feats.size(), feat_lengths))
        if feats.shape[0] == 0:
            continue
        optimizer.zero_grad()

        feats = feats.to(device)
        scripts = scripts.to(device)

        #src_len = scripts.size(1)
        target = scripts[:, 1:]

        #model.module.flatten_parameters()
        feat_lengths = torch.LongTensor(feat_lengths)
        #리스트가 만들어짐 문장길이x배치사이즈x사전길이
        logit, _ = model(x=(feats, feat_lengths))
        #logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=teacher_forcing_ratio)
        # 배치사이즈로 연결함 문장길이x배치사이즈x사전길이 -> 배치사이즈x문장길이x사전길이
        logit = torch.stack(logit, dim=1).to(device)

        y_hat = logit.max(-1)[1]

        y_hat = decoding_greedy(y_hat)

        #모든 글자를 일차원으로 나열하고 그 글자에대한 확률값을 에러율로 사용 - CrossEntropy
        #loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))

        #CTC Loss function 적용
        input_lengths = torch.full(size=(scripts.size(0),), fill_value=logit.size(1), dtype=torch.long)
        target_lengths = torch.LongTensor(script_lengths)- 1
        loss = criterion(logit.contiguous().transpose(0,1), target.contiguous(),input_lengths, target_lengths)
        #logger.info("loss : %s " % loss.item())
        total_loss += loss.item()
        total_num += sum(feat_lengths)

        display = random.randrange(0, 100) == 0
        #logger.info("predict : %s " % y_hat)
        #logger.info("target : %s " % target)
        dist, length = get_distance(target, y_hat, display=display)
        total_dist += dist
        total_length += length

        total_sent_num += target.size(0)

        loss.mean().backward()
        optimizer.step()

        if batch % print_batch == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            logger.info('batch: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'
                .format(batch,
                        #len(dataloader),
                        total_batch_size,
                        total_loss / total_num,
                        total_dist / total_length,
                        elapsed, epoch_elapsed, train_elapsed))
            logger.info("predict : %s" % label_to_string(y_hat[0]))
            logger.info(" target : %s" % label_to_string(target[0]))
            begin = time.time()

            nsml.report(False,
                        step=train.cumulative_batch_count, train_step__loss=total_loss/total_num,
                        train_step__cer=total_dist/total_length)
        batch += 1
        train.cumulative_batch_count += 1

    logger.info('train() completed')
    return total_loss / total_num, total_dist / total_length


train.cumulative_batch_count = 0


def evaluate(model, queue, criterion, device):
    logger.info('evaluate() start')
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0

    model.eval()

    with torch.no_grad():
        for feats, scripts, feat_lengths, script_lengths in queue:
            if feats.shape[0] == 0:
                continue

            feats = feats.to(device)
            scripts = scripts.to(device)

            src_len = scripts.size(1)
            target = scripts[:, 1:]

            #model.module.flatten_parameters()
            logit, _ = model(feats, feat_lengths, scripts, teacher_forcing_ratio=0.0)

            logit = torch.stack(logit, dim=1).to(device)

            y_hat = logit.max(-1)[1]
            y_hat = decoding_greedy(y_hat)
            #loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
            # CTC Loss function 적용
            input_lengths = torch.full(size=(scripts.size(0),), fill_value=logit.size(1), dtype=torch.long)
            target_lengths = torch.LongTensor(script_lengths) - 1
            loss = criterion(logit.contiguous().transpose(0, 1), target.contiguous(), input_lengths, target_lengths)
            total_loss += loss.item()
            total_num += sum(feat_lengths)

            display = random.randrange(0, 100) == 0
            dist, length = get_distance(target, y_hat, display=display)
            total_dist += dist
            total_length += length
            total_sent_num += target.size(0)

    logger.info('evaluate() completed')
    return total_loss / total_num, total_dist / total_length

def bind_model(model, optimizer=None):
    def load(filename, **kwargs):
        state = torch.load(os.path.join(filename, 'model.pt'))
        model.load_state_dict(state['model'])
        #model2.load_state_dict(state['model2'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    def save(filename, **kwargs):
        state = {
            'model': model.state_dict(),
            #'model2': model2.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(filename, 'model.pt'))
    #모델 제출시 실행되는 코드
    def infer(wav_path):
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input = get_spectrogram_feature(wav_path).unsqueeze(0)
        input = input.to(device)

        logit, _ = model(input_variable=input, input_lengths=None, teacher_forcing_ratio=0)
        logit = torch.stack(logit, dim=1).to(device)
        #beam search - language model과 혼용
        #y_hat = list()
        #for data in logit:
        #    y_hat.append(beam_search_decoder(data,3)[0])
        y_hat = logit.max(-1)[1] # greedy search  배치사이즈x문장의글자수
        y_hat = decoding_greedy(y_hat)
        hyp = label_to_string(y_hat)

        return hyp[0]

    nsml.bind(save=save, load=load, infer=infer) # 'nsml.bind' function must be called at the end.

def split_dataset(config, wav_paths, script_paths, valid_ratio=0.05):
    train_loader_count = config.workers
    records_num = len(wav_paths)
    batch_num = math.ceil(records_num / config.batch_size)

    valid_batch_num = math.ceil(batch_num * valid_ratio)
    train_batch_num = batch_num - valid_batch_num

    batch_num_per_train_loader = math.ceil(train_batch_num / config.workers)

    train_begin = 0
    train_end_raw_id = 0
    train_dataset_list = list()

    for i in range(config.workers):

        train_end = min(train_begin + batch_num_per_train_loader, train_batch_num)

        train_begin_raw_id = train_begin * config.batch_size
        train_end_raw_id = train_end * config.batch_size

        train_dataset_list.append(BaseDataset(
                                        wav_paths[train_begin_raw_id:train_end_raw_id],
                                        script_paths[train_begin_raw_id:train_end_raw_id],
                                        SOS_token, EOS_token))
        train_begin = train_end 

    valid_dataset = BaseDataset(wav_paths[train_end_raw_id:], script_paths[train_end_raw_id:], SOS_token, EOS_token)

    return train_batch_num, train_dataset_list, valid_dataset

def main():

    global char2index
    global index2char
    global SOS_token
    global EOS_token
    global PAD_token

    parser = argparse.ArgumentParser(description='Speech hackathon Baseline')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size of model (default: 256)')
    parser.add_argument('--layer_size', type=int, default=3, help='number of layers of model (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate in training (default: 0.2)')
    parser.add_argument('--bidirectional', action='store_true', help='use bidirectional RNN for encoder (default: False)')
    parser.add_argument('--use_attention', action='store_true', default=True,help='use attention between encoder-decoder (default: False)')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size in training (default: 32)')
    parser.add_argument('--workers', type=int, default=4, help='number of workers in dataset loader (default: 4)')
    parser.add_argument('--max_epochs', type=int, default=10, help='number of max epochs in training (default: 10)')
    parser.add_argument('--lr', type=float, default=5e-04, help='learning rate (default: 0.0001)')
    parser.add_argument('--teacher_forcing', type=float, default=0.5, help='teacher forcing ratio in decoder (default: 0.5)')
    parser.add_argument('--max_len', type=int, default=80, help='maximum characters of sentence (default: 80)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save_name', type=str, default='model', help='the name of model in nsml or local')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument("--pause", type=int, default=0)

    args = parser.parse_args()

    char2index, index2char = label_loader.load_label('./hackathon.labels')
    jasper_model_definition['labels']['labels'] = index2char
    SOS_token = char2index['<s>']
    EOS_token = char2index['</s>']
    PAD_token = char2index['_']

    #bs.testBeamSearch(index2char, "휴무일")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    featurizer_config = jasper_model_definition['input']

    # N_FFT: defined in loader.py
    feature_size = N_FFT / 2 + 1
    model = Jasper(feature_config=featurizer_config, jasper_model_definition=jasper_model_definition, feat_in=1024,
                   num_classes=len(index2char))

    #enc = EncoderRNN(feature_size, args.hidden_size,
    #                 input_dropout_p=args.dropout, dropout_p=args.dropout,
    #                 n_layers=args.layer_size, bidirectional=args.bidirectional, rnn_cell='lstm', variable_lengths=False)
    #
    #dec = DecoderRNN(len(char2index), args.max_len, args.hidden_size * (2 if args.bidirectional else 1),
    #                 SOS_token, EOS_token,
    #                 n_layers=args.layer_size, rnn_cell='lstm', bidirectional=args.bidirectional,
    #                 input_dropout_p=args.dropout, dropout_p=args.dropout, use_attention=args.use_attention)
    #
    #model = Seq2seq(enc, dec)
    #model.flatten_parameters()


    for param in model.parameters():
        param.data.uniform_(-0.08, 0.08)

    model = nn.DataParallel(model).to(device)
    #Cross Entropy 적용
    #optimizer = optim.Adam(model.module.parameters(), lr=args.lr)
    #criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)
    #CTCLoss 적용 Adam->SGD
    optimizer = optim.Adam(model.module.parameters(), lr=args.lr) #optim.SGD(model.module.parameters(), lr=args.lr)#
    criterion = nn.CTCLoss(reduction='sum', blank=PAD_token,zero_infinity=True).to(device)

    #파일남아있는지 확인용 코드


    bind_model(model, optimizer)

    if args.pause == 1:
        nsml.paused(scope=locals())

    if args.mode != "train":
        return

    #try:
    #    nsml.load(checkpoint='75_model', session='team138/sr-hack-2019-dataset/151')
    #except Exception:
    #    pass

    data_list = os.path.join(DATASET_PATH, 'train_data', 'data_list.csv')
    wav_paths = list()
    script_paths = list()

    with open(data_list, 'r') as f:
        for line in f:
            # line: "aaa.wav,aaa.label"

            wav_path, script_path = line.strip().split(',')
            wav_paths.append(os.path.join(DATASET_PATH, 'train_data', wav_path))
            script_paths.append(os.path.join(DATASET_PATH, 'train_data', script_path))

    best_loss = 1e10
    best_cer = 1e10
    begin_epoch = 0

    # load all target scripts for reducing disk i/o
    target_path = os.path.join(DATASET_PATH, 'train_label')
    load_targets(target_path)

    train_batch_num, train_dataset_list, valid_dataset = split_dataset(args, wav_paths, script_paths, valid_ratio=0.05)
    logger.info('wav to melspectrogram in train set')

    #log - melspectrogram 데이터 로드 학습데이터
    batch_idx = 0

    all_train_data = list()
    train_data = list()
    for train_dataset in train_dataset_list:

        for idx, _ in enumerate(train_dataset.wav_paths):
            train_data.append(train_dataset.getitem(idx))
            batch_idx+=1
            if batch_idx % args.batch_size == 0:
                random.shuffle(train_data)
                batch = _collate_fn(train_data)
                all_train_data.append(batch)
                train_data = list()
    if len(train_data) > 0:
        random.shuffle(train_data)
        batch = _collate_fn(train_data)
        all_train_data.append(batch)

    logger.info('wav to melspectrogram in validation set')
    #log - melspectrogram 데이터 로드 테스트데이터
    all_valid_data = list()
    valid_data = list()
    batch_idx = 0
    for idx, _ in enumerate(valid_dataset.wav_paths):
        valid_data.append(valid_dataset.getitem(idx))
        batch_idx += 1
        if batch_idx % args.batch_size == 0:
            batch = _collate_fn(valid_data)
            all_valid_data.append(batch)
            valid_data = list()
    if len(valid_data) > 0:
        batch = _collate_fn(valid_data)
        all_valid_data.append(batch)

    logger.info('start')

    train_begin = time.time()

    for epoch in range(begin_epoch, args.max_epochs):
        random.shuffle(all_train_data)
        #Queue 생성 및 만들어진 데이터 저장
        #train_queue = queue.Queue(args.workers * 2)

        #train_loader = MultiLoader(train_dataset_list, train_queue, args.batch_size, args.workers)
        #train_loader.start()


        train_loss, train_cer = train(model, train_batch_num, all_train_data, criterion, optimizer, device, train_begin, args.workers, 10, args.teacher_forcing)
        logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

        #train_loader.join()

        #valid_queue = queue.Queue(args.workers * 2)
        #valid_loader = BaseDataLoader(valid_dataset, valid_queue, args.batch_size, 0)
        #valid_loader.start()

        eval_loss, eval_cer = evaluate(model, all_valid_data, criterion, device)
        logger.info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, eval_loss, eval_cer))

        #valid_loader.join()

        nsml.report(False,
            step=epoch, train_epoch__loss=train_loss, train_epoch__cer=train_cer,
            eval__loss=eval_loss, eval__cer=eval_cer)

        best_model = (eval_loss < best_loss)
        best_model_cer = (eval_cer < best_cer)

        if epoch % 5 == 0:
            nsml.save("%s_model" % epoch)
        if best_model:
            nsml.save('best_loss')
            best_loss = eval_loss
        if best_model_cer:
            nsml.save('best_cer')
            best_model_cer = eval_cer
if __name__ == "__main__":
    main()
