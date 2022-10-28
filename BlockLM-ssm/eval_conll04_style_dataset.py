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
from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence
from SwissArmyTransformer.generation.sampling_strategies import BeamSearchStrategy, BaseStrategy, \
    ConstrainedBeamSearchStrategy
from SwissArmyTransformer.generation.utils import timed_name, generate_continually


class NewKGEvaluator(object):
    eval_relation = "tonality"
    prompts = {
        "terminus location": "[X]'s terminus location is [MASK]",
        "tonality": "[X]'s musical tonality is [MASK]"
    }

    def __init__(self, process):
        base_path = "/dataset/fd5061f6/hezhenyu/code_data/nell2_data_mix/"
        relation = self.eval_relation.replace('/', '<>').replace(' ', '_')
        folders = list(os.listdir(base_path))
        find_relation = False
        for f in folders:
            if f.startswith(relation):
                relation = f
                find_relation = True
                break
        if not find_relation:
            raise RuntimeError(relation + " was not found in data direcotry " + base_path)
        self.source = open(
            base_path + '{}/test.source'.format(relation))
        self.target = open(
            base_path + '{}/test.target'.format(relation))

        self.process = process

    def run_test(self):
        num_of_correct, num_of_gt, num_of_pred = 0, 0, 0
        num_of_pred_not_none, num_of_pred_not_none_correct, num_of_not_none, num_of_not_none_pred_correct = 0, 0, 0, 0
        for idx, (src, tgt) in tqdm(enumerate(list(zip(self.source, self.target)))):
            src, tgt = src.strip('\n'), tgt.strip('\n')
            num_of_gt += 1
            tokens = src.replace("Please answer", "Please summarize")

            tokens = self.prompts[self.eval_relation].replace('[X]', tokens)
            pred = self.process(tokens)[0]
            pred = pred.split(tokens.replace(' [MASK]', '').replace('[X]', ''))[-1].strip()
            # pred = pred.replace(tokens.replace(' [MASK]', ''), '').strip(' ').replace('[CLS]  ', '')
            if pred in {'the', 'a', ':', 'of', 'in', 'on', '', ' ', '"'}:
                pred = 'None'
            if pred.startswith('the '):
                pred = pred[4:]
            if pred == tgt:
                num_of_correct += 1
            if tgt != "None":
                num_of_not_none += 1
                if pred == tgt:
                    num_of_not_none_pred_correct += 1
            if pred != "None":
                num_of_pred_not_none += 1
                if pred == tgt:
                    num_of_pred_not_none_correct += 1
            print("{} Pred: {} Ground: {}".format(idx, pred, tgt))
        print("EM: {} Not_none_prec: {} Not_none_recall: {}".format(num_of_correct / num_of_gt,
                                                                    num_of_not_none_pred_correct / num_of_not_none,
                                                                    num_of_pred_not_none_correct / num_of_pred_not_none))


class CoNLL04Evaluator(object):
    relation_mapping = {
        "OrgBased_In": "is based in",
        "Located_In": "is located in",
        "Live_In": "lived in",
        "Work_For": "works for",
        "Kill": "killed"
    }

    allowed_relations = {
        'Org': ['OrgBased_In'],
        'Other': [],
        'Loc': ['Located_In'],
        'Peop': ['Live_In', 'Work_For', 'Kill']
    }

    def __init__(self, process):
        self.raw_data = json.load(open('/dataset/fd5061f6/liuxiao/data/conll04/conll04_test.json'))
        self.dataset = []
        self.preprocess()

        self.process = process

    def preprocess(self):
        for doc in self.raw_data:
            text = ' '.join(doc['tokens'])
            entities = [(' '.join(doc['tokens'][e['start']:e['end']]), e['type']) for e in doc['entities']]
            relations = [(entities[r['head']][0], r['type'], entities[r['tail']][0]) for r in doc['relations']]
            self.dataset.append({
                "text": text,
                "entities": entities,
                "relations": relations,
                "id": doc['orig_id']
            })

    def run_test(self):
        num_of_correct, num_of_gt, num_of_pred = 0, 0, 0
        per_rel = {'OrgBased_In': [0, 0, 0], 'Located_In': [0, 0, 0], 'Live_In': [0, 0, 0], 'Work_For': [0, 0, 0],
                   'Kill': [0, 0, 0]}
        for doc in tqdm(self.dataset):
            preds = []
            for ent in doc['entities']:
                for rel in self.allowed_relations[ent[1]]:
                    tokens = "{} Please summarize: {} {} [MASK]".format(doc['text'], ent[0], self.relation_mapping[rel])
                    txts = self.process(tokens)
                    pred = txts[0].replace(tokens.replace(' [MASK]', ''), '').replace(' [CLS]  ', '')
                    if pred in {'the', 'a', ':'}:
                        continue
                    if pred == ent[0]:
                        continue
                    if pred.startswith('the '):
                        pred = pred[4:]
                    preds.append((ent[0], rel, pred))
            for triple in preds:
                num_of_pred += 1
                per_rel[triple[1]][2] += 1
            for triple in doc['relations']:
                num_of_gt += 1
                per_rel[triple[1]][1] += 1
                for pred in preds:
                    if triple[0] == pred[0] and triple[1] == pred[1] and pred[2] == triple[2]:
                        num_of_correct += 1
                        per_rel[triple[1]][0] += 1
            doc['preds'] = preds
        json.dump(self.dataset, open('conll04_dump.json', 'w'), indent=4, ensure_ascii=False)
        json.dump(per_rel, open('conll04_per_rel_dump.json', 'w'), indent=4, ensure_ascii=False)
        print(num_of_correct, num_of_gt, num_of_pred)


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


def main(args):
    args.do_train = False
    initialize_distributed(args)
    tokenizer = get_tokenizer(args)
    # build model
    model = GLMFPrefixModel(args) if args.prefix_prompt > 0 else GLMModel(args)
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    if args.fp16:
        model = model.half()
    model = model.to(args.device)
    load_pretrained(model, args.load_pretrained, args)
    set_random_seed(args.seed)
    model.eval()

    end_tokens = [tokenizer.get_command('eop').Id, tokenizer.get_command('eos').Id]
    # define function for each query
    if args.num_beams == 1:
        strategy = BaseStrategy(temperature=args.temperature, top_k=args.top_k, end_tokens=end_tokens)
    else:
        strategy = ConstrainedBeamSearchStrategy(args.num_beams, length_penalty=args.length_penalty, consider_end=True,
                                                 end_tokens=end_tokens, no_repeat_ngram_size=args.no_repeat_ngram_size,
                                                 min_tgt_length=args.min_tgt_length, tokenizer=tokenizer)

    def process(raw_text):
        strategy.refresh()
        if args.with_id:
            query_id, raw_text = raw_text.split('\t')
        # add MASK
        generation_mask = '[gMASK]' if args.task_mask else '[MASK]'
        if 'MASK]' not in raw_text:
            raw_text += ' ' + generation_mask
        seq = tokenizer.EncodeAsIds(raw_text).tokenization
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

        # save
        if args.with_id:
            full_path = os.path.join(args.output_path, query_id + '.txt')
        else:
            prefix = raw_text.replace('/', '')[:20]
            full_path = timed_name(prefix, '.txt', args.output_path)
            # print(txts[0])  # print the first.
        with open(full_path, 'w') as fout:
            for txt in txts:
                fout.write(txt + '\n')
        os.chmod(full_path, stat.S_IRWXO + stat.S_IRWXG + stat.S_IRWXU)

        return txts

    os.makedirs(args.output_path, exist_ok=True)
    # generate_continually(process, args.input_source)
    evaluator = NewKGEvaluator(process)
    evaluator.run_test()


if __name__ == "__main__":
    args = get_args()

    with torch.no_grad():
        main(args)
