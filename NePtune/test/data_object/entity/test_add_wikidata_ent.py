from data_object import WikipediaEntity, WikidataEntity, WikipediaPage, BaseParagraph, BaseSentence

from os.path import join
from tqdm import tqdm
from urllib.parse import unquote
from collections import defaultdict

import json
import re

BASE_DIR = '/raid/liuxiao/nell_data/hailong'


def get_entity_alias_description():
    res, last_id = {"text": None, "description": None, "aliases": [], 'sourceId': 'Q31'}, 'Q31'
    for line in open(join(BASE_DIR, 'ins-label-alias.txt')):
        qid, prop, label = tuple(line.strip('\n').split('\t\t'))
        if qid != last_id:
            yield last_id, res
            last_id = qid
            res = {"text": None, "description": None, "aliases": [], 'sourceId': qid}
        if prop == 'schema:description':
            res['description'] = label[1:-4]
        elif prop == "rdfs:label":
            res['text'] = label[1:-4]
        elif prop == 'skos:altLabel':
            res['aliases'].append(label[1:-4])
        else:
            raise NotImplementedError(f"{prop} unknown.")


def clean(_url):
    pat = re.compile(r'<https://en.wikipedia.org/wiki/(.*?)>')
    return unquote(pat.search(_url).group(1))


def add_wikidata_ent_to_db():
    # get cross link
    qid2title = json.load(open('/raid/liuxiao/nell_data/enwiki/qid_to_underline_title.json'))
    entity_label_to_objectId = json.load(open('/raid/liuxiao/nell_data/enwiki/entity_label_to_objectid.json'))

    for qid, doc in tqdm(get_entity_alias_description()):
        title = qid2title.get(qid)
        if title is not None:
            entity_objectId = entity_label_to_objectId.get(title)
            if entity_objectId is not None:
                entity = WikipediaEntity.objects.get(id=entity_objectId)

                try:
                    # save wikidata entity
                    doc['equivalents'] = {"wikipedia": entity}
                    wikidata_entity = WikidataEntity(**doc)
                    wikidata_entity.save()

                    # save wikipedia entity
                    entity.equivalents = {"wikidata": wikidata_entity}
                    entity.save()
                except:
                    pass

                continue
        try:
            wikidata_entity = WikidataEntity(**doc)
            wikidata_entity.save()
        except:
            print(qid)


def add_wikidata_ent_type_to_db():
    qid2types = json.load(open('/raid/liuxiao/nell_data/wikidata/wikidata_entity_types_from_constraints.json'))
    for entity in tqdm(WikidataEntity.objects.no_cache()):
        doc = qid2types.get(entity.sourceId)
        if doc is None:
            continue
        type_doc = dict()
        if len(doc['as_head']) > 0:
            type_doc["property-as_head"] = doc['as_head']
        if len(doc['as_tail']) > 0:
            type_doc["property-as_tail"] = doc['as_tail']
        if len(type_doc) > 0:
            entity.types = type_doc
            entity.save()


def get_missing_tail_entities():
    qid2title = json.load(open('/raid/liuxiao/nell_data/enwiki/qid_to_underline_title.json'))
    missing_qid = set()
    for line in tqdm(open('/raid/liuxiao/nell_data/hailong/ins-relation-triple.txt')):
        head, prop, tail = tuple(line.strip('\n').split('\t\t'))
        if tail not in qid2title:
            missing_qid.add(tail)
    missing_qid = list(missing_qid)
    json.dump(missing_qid, open('/raid/liuxiao/nell_data/wikidata/missing_qid_as_tail.json', 'w'))
    print("Missing:", len(missing_qid))


def add_other_wikidata_entity():
    for line_idx, line in tqdm(enumerate(open('/raid/liuxiao/nell_data/wikidata/missing_qid_as_tail_docs.jsonl'))):
        if line_idx < 93:
            continue
        doc = json.loads(line)
        if doc['text'] is None:
            continue
        entity = WikidataEntity(**doc)
        entity.save()


if __name__ == '__main__':
    add_other_wikidata_entity()
