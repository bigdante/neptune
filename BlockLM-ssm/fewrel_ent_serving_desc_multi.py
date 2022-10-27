# -*- encoding: utf-8 -*-
'''
@File    :   inference_glm.py
@Time    :   2021/10/22 19:41:58
@Author  :   Ming Ding
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import json
from tqdm import tqdm
import re

import os
import sys
import random
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import argparse
import stat
from functools import partial

from arguments import get_args
from model import GLMFPrefixModel
from SwissArmyTransformer import mpu, get_tokenizer, initialize_distributed, set_random_seed
from finetune_glm import load_pretrained
from SwissArmyTransformer.model import GLMModel
from SwissArmyTransformer.model.mixins import CachedAutoregressiveMixin
from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence, filling_batch_sequence
from SwissArmyTransformer.generation.sampling_strategies import BeamSearchStrategy, BaseStrategy, \
    ConstrainedBeamSearchStrategy, ConstrainedBaseStrategy
from SwissArmyTransformer.generation.utils import timed_name, generate_continually

from flask import Flask, request, jsonify


def extract_answer(inputs, targets):
    template = re.escape(inputs).replace('\\[MASK\\]', '(.+)')
    res = re.search(template, targets)
    if res is None:
        return ""
    return res.group(1)


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        return json.JSONEncoder.default(self, o)


encoder = JSONEncoder()


def output_process(result):
    result = json.loads(encoder.encode(result))
    return jsonify(result)


class Server(object):
    def __init__(self, process, tokenizer):
        self.process = process
        self.tokenizer = tokenizer
        pass

    def predict(self, queries, mbz=5):
        start = datetime.now()
        answers = []
        tokens = self.tokenizer.EncodeAsIds(queries[0])
        if len(tokens) >= 350:
            mbz = 1
        elif len(tokens) >= 280:
            mbz = 2
        elif len(tokens) >= 230:
            mbz = 3
        elif len(tokens) >= 200:
            mbz = 4
        elif len(tokens) >= 150:
            mbz = 5
        else:
            mbz = 6
        for idx in range((len(queries) + mbz - 1) // mbz):
            query = queries[idx * mbz: min((idx + 1) * mbz, len(queries))]
            try:
                answers.extend(self.process(query))
            except:
                answers.extend([''] * len(query))
        print("Time elapse:", datetime.now() - start)
        return answers


def get_masks_and_position_ids_glm(seq, mask_position, context_length):
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., :context_length] = 1
    attention_mask.unsqueeze_(1)

    position_ids = torch.zeros(2, len(seq), device=tokens.device, dtype=torch.long)
    torch.arange(0, context_length, out=position_ids[0, :context_length])
    position_ids[0, context_length:] = mask_position
    torch.arange(1, len(seq) - context_length + 1, out=position_ids[1, context_length:])

    position_ids = position_ids.unsqueeze(0)
    return tokens, attention_mask, position_ids


def get_masks_and_position_ids_glm_batch(seq, pad_num, mask_position, context_length):
    tokens = seq
    bz, seq_len = seq.shape[0], seq.shape[1]

    attention_mask = torch.ones((bz, seq_len, seq_len), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., :context_length] = 1
    for sample_idx, pad in enumerate(pad_num):
        attention_mask[sample_idx, :, :pad] = 0
        attention_mask[sample_idx, :pad, :] = 0
    attention_mask.unsqueeze_(1)

    position_ids = torch.zeros((bz, 2, seq_len), device=tokens.device, dtype=torch.long)
    for sample_idx, pad in enumerate(pad_num):
        torch.arange(0, context_length - pad, out=position_ids[sample_idx, 0, pad:context_length])
        position_ids[sample_idx, 0, context_length:] = mask_position - pad
        torch.arange(1, seq_len - context_length + 1, out=position_ids[sample_idx, 1, context_length:])

    return tokens, attention_mask, position_ids


def main(args):
    args.do_train = False
    initialize_distributed(args)
    tokenizer = get_tokenizer(args)
    # build model
    print('build model...')
    model = GLMFPrefixModel(args) if args.prefix_prompt > 0 else GLMModel(args)
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    if args.fp16:
        model = model.half()
    model = model.to(args.device)
    load_pretrained(model, args.load_pretrained, args)
    set_random_seed(args.seed)
    model.eval()
    print('\tdone.')

    end_tokens = [tokenizer.get_command('eop').Id, tokenizer.get_command('eos').Id]
    # define function for each query
    if args.num_beams == 1:
        if args.inference_strategy_constrained:
            strategy = ConstrainedBaseStrategy(temperature=args.temperature, top_k=args.top_k, end_tokens=end_tokens,
                                               tokenizer=tokenizer)
        else:
            strategy = BaseStrategy(temperature=args.temperature, top_k=args.top_k, end_tokens=end_tokens)
    else:
        if args.inference_strategy_constrained:
            strategy = ConstrainedBeamSearchStrategy(args.num_beams, length_penalty=args.length_penalty,
                                                     consider_end=True, end_tokens=end_tokens,
                                                     no_repeat_ngram_size=args.no_repeat_ngram_size,
                                                     min_tgt_length=args.min_tgt_length, tokenizer=tokenizer)
        else:
            strategy = BeamSearchStrategy(args.num_beams, length_penalty=args.length_penalty, consider_end=True,
                                          end_tokens=end_tokens, no_repeat_ngram_size=args.no_repeat_ngram_size,
                                          min_tgt_length=args.min_tgt_length)

    def process_batch(raw_texts):
        '''
        raw_texts should share the same context.
        '''
        # torch.cuda.empty_cache()
        if args.inference_strategy_constrained:
            strategy.refresh()
        seqs, max_seq_len = [], 0
        for raw_text in raw_texts:
            tokens = [tokenizer.get_command('ENC').Id] + tokenizer.EncodeAsIds(raw_text).tokenization
            seqs.append(tokens)
            max_seq_len = max(len(tokens), max_seq_len)
        print("Sequence length:", max_seq_len)
        padded_seqs, num_of_pads = [], []
        for tokens in seqs:
            num_of_pad = max_seq_len - len(tokens)
            num_of_pads.append(num_of_pad)
            padded_seqs.append([tokenizer.get_command('ENC').Id] * num_of_pad + tokens)

        # set context constraints
        context_seq = tokenizer.EncodeAsIds(raw_texts[0].split('\n\n')[1]).tokenization
        strategy.context_tokens = torch.cuda.LongTensor([context_seq], device=args.device)

        # get masked positions
        mask_position = len(padded_seqs[0]) - 1
        get_func = partial(get_masks_and_position_ids_glm_batch,
                           pad_num=num_of_pads,
                           mask_position=mask_position,
                           context_length=max_seq_len)
        output_list = []

        # for tim in range(max(len(padded_seqs) // mbz, 1)):
        input_seq = torch.cuda.LongTensor(
            [(seq + [tokenizer.get_command('sop').Id] + [-1] * 20) for seq in
             padded_seqs],
            device=args.device)

        output = filling_batch_sequence(model, input_seq, strategy=strategy,
                                        get_masks_and_position_ids=get_func)[0]  # we don't use mems, fill back
        # if isinstance(output, torch.Tensor):  # different strategies
        #     output = list(output)

        output_list.extend(output.split(1))

        for i in range(len(output_list)):
            output = output_list[i].tolist()[0]
            try:
                unfinished = output.index(-1)
            except ValueError:
                unfinished = len(output)
            if output[unfinished - 1] in end_tokens:
                unfinished -= 1
            bog = output.index(tokenizer.get_command('sop').Id)
            output_list[i] = output[bog + 1: unfinished]

        txts = []
        for seq in output_list:
            decode_tokens = tokenizer.DecodeIds(seq)
            txts.append(decode_tokens.split(' <|endofpiece|>')[0].strip())

        return txts

    def process(raw_text):
        if args.inference_strategy_constrained:
            strategy.refresh()
        if args.with_id:
            query_id, raw_text = raw_text.split('\t')
        # add MASK
        generation_mask = '[gMASK]' if args.task_mask else '[MASK]'
        if 'MASK]' not in raw_text:
            raw_text += ' ' + generation_mask
        seq = tokenizer.EncodeAsIds(raw_text).tokenization
        context_seq = tokenizer.EncodeAsIds(raw_text.split('\n\n')[1]).tokenization
        strategy.context_tokens = torch.cuda.LongTensor([context_seq], device=args.device)
        seq = [tokenizer.get_command('ENC').Id] + seq
        if not raw_text.endswith('MASK]'):
            seq = seq + [tokenizer.get_command('eos').Id]
        # print('raw text: {}\n'.format(raw_text))
        if len(seq) > args.max_sequence_length:
            raise ValueError('text too long.')

        # generation
        mbz = args.max_inference_batch_size
        assert args.batch_size < mbz or args.batch_size % mbz == 0
        output_list = [seq]
        # continually detect the first mark position
        while True:
            seq = output_list[0]  # TODO find the best one
            # detect
            mask_tokens = ['MASK', 'sMASK', 'gMASK'] if args.task_mask else ['MASK']
            mask_tokens = [tokenizer.get_command(token).Id for token in mask_tokens]
            mask_position = len(seq)
            for token in mask_tokens:
                try:
                    mask_position = min(mask_position, seq.index(token))
                except ValueError:
                    pass
            if mask_position == len(seq):
                break

            get_func = partial(get_masks_and_position_ids_glm, mask_position=mask_position, context_length=len(seq))
            output_list = []
            for tim in range(max(args.batch_size // mbz, 1)):
                input_seq = torch.cuda.LongTensor(
                    seq + [tokenizer.get_command('sop').Id] + [-1] * (args.out_seq_length - len(seq) - 1),
                    device=args.device)
                output = filling_sequence(model, input_seq,
                                          batch_size=args.num_beams,
                                          strategy=strategy,
                                          log_attention_weights=None,
                                          get_masks_and_position_ids=get_func
                                          )[0]  # we don't use mems, fill back
                if isinstance(output, torch.Tensor):  # different strategies
                    output = list(output)

                output_list.extend(output)

            # clip -1s and fill back generated things into seq
            for i in range(len(output_list)):
                output = output_list[i].tolist()
                try:
                    unfinished = output.index(-1)
                except ValueError:
                    unfinished = len(output)
                if output[unfinished - 1] in end_tokens:
                    unfinished -= 1
                bog = output.index(tokenizer.get_command('sop').Id)
                output_list[i] = output[:mask_position] + output[bog + 1:unfinished] + output[mask_position + 1:bog]

        # decoding
        txts = []
        for seq in output_list:
            decode_tokens = tokenizer.DecodeIds(seq)
            txts.append(decode_tokens)

        return txts

    os.makedirs(args.output_path, exist_ok=True)
    # generate_continually(process, args.input_source)
    processor = Server(process_batch, tokenizer)

    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False

    @app.route('/query', methods=['POST'])
    def index():
        """
        data here should be in the form of a list [query1, ...]
        """
        queries = request.get_json()
        print("Num of queries:", len(queries))
        res = processor.predict(queries)
        # print(json.dumps(res, indent=4, ensure_ascii=False))
        return output_process(res)

    app.run(host="0.0.0.0", port=args.serving_port)


if __name__ == "__main__":
    args = get_args()

    with torch.no_grad():
        main(args)
