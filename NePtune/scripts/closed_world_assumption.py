import json
import sys

from tqdm import tqdm
from collections import defaultdict

from data_object import WikipediaEntity, BaseSentence, WikidataEntity, BaseRelation

BASE_DIR = 'extracted_facts/2022-04-12'


def get_head_qid():
    data = json.load(open(f'/raid/liuxiao/nell_data/{BASE_DIR}/neptune1.0_facts.json'))
    qids = set()
    for idx, doc in tqdm(enumerate(data)):
        # get sentence object
        evidence_id = doc['evidence']['_ref']['$id']['$oid']
        sentence = BaseSentence.objects.get(id=evidence_id)

        # get mention
        entity = None
        for mention in sentence.mentions:
            if mention['charSpan'][0] == doc['headSpan'][0] and mention['charSpan'][1] == doc['headSpan'][1]:
                if mention.entity is not None:
                    entity = mention.entity
                    break

        # assert entity is not None
        # get qid
        if len(entity.equivalents) == 0:
            continue
        qids.add(entity.equivalents['wikidata'].sourceId)

    json.dump(list(qids), open(f'/raid/liuxiao/nell_data/{BASE_DIR}/neptune1.0_facts_head_qids.json', 'w'))
    print(len(qids))


def get_gt_from_wikidata():
    qids = set(json.load(open(f'/raid/liuxiao/nell_data/{BASE_DIR}/neptune1.0_facts_head_qids.json')))
    triple = defaultdict(dict)
    for line in tqdm(open('/raid/liuxiao/nell_data/hailong/ins-relation-triple.txt')):
        head, prop, tail = tuple(line.strip('\n').split('\t\t'))
        if head in qids:
            if prop not in triple[head]:
                triple[head][prop] = []
            triple[head][prop].append(tail)
    json.dump(triple, open(f'/raid/liuxiao/nell_data/{BASE_DIR}/neptune1.0_closed_world_facts.json', 'w'))


def load_country_adj():
    country2adj = dict()
    for line in open('country_adj.tsv'):
        key, value = tuple(line.strip('\n').split('\t'))
        country2adj[key] = value
    json.dump(country2adj, open('country2adj.json', 'w'))


def get_labeled_sample():
    country2adj = json.load(open('country2adj.json'))
    triple = json.load(open(f'/raid/liuxiao/nell_data/{BASE_DIR}/neptune1.0_closed_world_facts.json'))
    data = json.load(open(f'/raid/liuxiao/nell_data/{BASE_DIR}/neptune1.0_facts.json'))
    pos, neg, unknown = 0, 0, 0
    for idx, doc in tqdm(enumerate(data)):
        relation = BaseRelation.objects.get(id=doc['relation']['$oid'])
        pid = relation.sourceId

        evidence_id = doc['evidence']['_ref']['$id']['$oid']
        sentence = BaseSentence.objects.get(id=evidence_id)

        # get entity
        entity = None
        for mention in sentence.mentions:
            if mention['charSpan'][0] == doc['headSpan'][0] and mention['charSpan'][1] == doc['headSpan'][1]:
                if mention.entity is not None:
                    entity = mention.entity
                    break

        if len(entity.equivalents) == 0:
            continue
        head_qid = entity.equivalents['wikidata'].sourceId

        if head_qid not in triple:
            continue
        if pid not in triple[head_qid]:
            # wikidata中不存在该property，标签unknown
            unknown += 1
            doc['cws_label'] = -1
        else:
            allowed_aliases = []
            for tail_qid in triple[head_qid][pid]:
                wk_ent = WikidataEntity.objects(sourceId=tail_qid).first()
                if wk_ent is not None:
                    allowed_aliases.extend([wk_ent.text] + wk_ent.aliases)
                    if wk_ent.text in country2adj:
                        allowed_aliases.append(country2adj[wk_ent.text])
            doc['allowed_aliases'] = allowed_aliases
            aliases_lower = [a.lower() for a in allowed_aliases]
            if doc['tail'].lower() in aliases_lower:
                pos += 1
                doc['cws_label'] = 1
            else:
                neg += 1
                doc['cws_label'] = 0

    json.dump(data, open(f'/raid/liuxiao/nell_data/{BASE_DIR}/neptune1.0_facts_cws_annotation.json', 'w'), indent=4,
              ensure_ascii=False)
    print(f"Pos: {pos}, Neg: {neg}, Unknown: {unknown}")


def get_negative_sample():
    data = json.load(open(f'/raid/liuxiao/nell_data/{BASE_DIR}/neptune1.0_facts_cws_annotation.json'))
    neg, pos = [], []
    for doc in data:
        if doc.get('cws_label') == 0:
            neg.append(doc)
        elif doc.get('cws_label') == 1:
            pos.append(doc)
    neg_rel, pos_rel = defaultdict(int), defaultdict(int)
    for doc in neg:
        neg_rel[doc['relationLabel']] += 1
    for doc in pos:
        pos_rel[doc['relationLabel']] += 1
    print(json.dumps(dict(sorted(neg_rel.items(), key=lambda x: x[1], reverse=True)), indent=4))
    print(json.dumps(dict(sorted(pos_rel.items(), key=lambda x: x[1], reverse=True)), indent=4))

    json.dump(neg, open(f'/raid/liuxiao/nell_data/{BASE_DIR}/neptune1.0_facts_cws_neg.json', 'w'), indent=4, ensure_ascii=False)
    json.dump(pos, open(f'/raid/liuxiao/nell_data/{BASE_DIR}/neptune1.0_facts_cws_pos.json', 'w'), indent=4, ensure_ascii=False)


if __name__ == '__main__':
    print("get head id")
    get_head_qid()
    print("get groundtruth wikidata")
    get_gt_from_wikidata()
    print("get labeld samples")
    get_labeled_sample()
    print("get neg/pos relation stats")
    get_negative_sample()
