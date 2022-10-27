from data_object import *
from tqdm import tqdm
from bson import ObjectId
from collections import OrderedDict

import json
import os

marked_num = 0


def get_marked_invalid_sentence():
    all_ids = set()

    print("Process invalid data from shanhe")
    base_path = '/raid/liuxiao/data/shanhe/liuxiao/NePtune1.0/log'
    for directory in tqdm(list(os.listdir(base_path))):
        if not os.path.exists(f'{base_path}/{directory}/invalid_facts.jsonl'):
            continue
        for line in open(f'{base_path}/{directory}/invalid_facts.jsonl'):
            doc = json.loads(line)
            _id = doc['evidence']['_ref']['$id']['$oid']
            if _id not in all_ids:
                all_ids.add(_id)
        print(len(all_ids))

    print("Process invalid data from baai")
    base_path = '/raid/liuxiao/NePtune1.0/log'
    for directory in tqdm(list(os.listdir(base_path))):
        if not os.path.exists(f'{base_path}/{directory}/invalid_facts.jsonl'):
            continue
        for line in open(f'{base_path}/{directory}/invalid_facts.jsonl'):
            try:
                doc = json.loads(line)
            except:
                continue
            _id = doc['evidence']['_ref']['$id']['$oid']
            if _id not in all_ids:
                all_ids.add(_id)
        print(len(all_ids))

    with open('/raid/liuxiao/data/marked_sentence_id.txt', 'w') as f:
        for _id in all_ids:
            f.write(_id + '\n')


def get_marked_valid_sentence_shanhe():
    for idx, doc in tqdm(enumerate(TripleFact.objects.no_cache()[30469061 + 2139726 + 22577627 + 31939907:])):
        if 'has_prediction' not in doc.evidence.temp:
            doc.evidence.temp['has_prediction'] = True
            doc.evidence.save()


def add_processed_sentence_to_db():
    for idx, line in tqdm(enumerate(open('/raid/liuxiao/data/valid_sentence_id_shanhe.txt'))):
        _id = line.strip('\n')
        # print(_id)
        sentence = BaseSentence.objects.get(id=_id)
        sentence.temp['has_prediction'] = True
        sentence.save()


def get_marked_valid_sentence():
    sid = set()
    for idx, doc in tqdm(enumerate(TripleFact.objects.no_cache())):
        sid.add(json.loads(doc.to_json())['evidence']['_ref']['$id']['$oid'])

    with open('/raid/liuxiao/data/valid_sentence_id_shanhe.txt', 'w') as f:
        for _id in sid:
            f.write(_id + '\n')


def fine_unpredicted_sentence():
    out = open('/raid/liuxiao/data/invalid_blocks.txt', 'a')
    span, last_id = 0, None
    all_unprocessed = 0
    for sentence in tqdm(BaseSentence.objects.no_cache()):
        if last_id is not None and (sentence.temp.get('last_processed') or sentence.temp.get('has_prediction')):
            out.write(f'{last_id}\t{span}\n')
            last_id = None
            span = 0
            continue
        else:
            if span == 0:
                last_id = str(sentence.id)
            all_unprocessed += 1
            span += 1
    print("All unprocessed:", all_unprocessed)


def get_unrunning_sentence():
    counts = list()
    for line in open('/raid/liuxiao/data/sorted_invalid_blocks_gt_10.txt'):
        line = line.strip().split('\t')
        sentence = BaseSentence.objects.get(id=line[0])
        if sentence.temp.get('last_processed') or sentence.temp.get('has_prediction'):
            continue
        counts.append((line[0], int(line[1])))
    json.dump(counts, open('/raid/liuxiao/data/sorted_invalid_blocks_gt_10.json', 'w'), indent=4)


def mark_abs_extracted_sentence():
    num = len(list(os.listdir('/raid/liuxiao/NePtune1.0/log')))
    for idx, directory in enumerate(os.listdir('/raid/liuxiao/NePtune1.0/log')):
        if directory.startswith('2022'):
            print(f'{idx}/{num}', directory)
            for line in tqdm(open(f'/raid/liuxiao/NePtune1.0/log/{directory}/invalid_facts.jsonl')):
                doc = json.loads(line)
                sentence = BaseSentence.objects.get(id=doc['evidence']['_ref']['$id']['$oid'])
                sentence.temp['abs_extracted'] = True
                sentence.save()


if __name__ == '__main__':
    # get_unrunning_sentence()
    # get_marked_valid_sentence()
    # get_marked_valid_sentence_shanhe()
    mark_abs_extracted_sentence()
