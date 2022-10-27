from data_object import WikipediaEntity, WikipediaPage, BaseParagraph, BaseSentence

from os.path import join
from tqdm import tqdm
from urllib.parse import unquote
from collections import defaultdict

import json
import re

BASE_DIR = '/raid/liuxiao/nell_data/hailong'


def get_entity_name_to_objectId():
    entity_label_to_objectId = dict()
    for entity in tqdm(WikipediaEntity.objects.no_cache().as_pymongo()):
        if entity['text'] not in entity_label_to_objectId:
            entity_label_to_objectId[entity['text']] = str(entity['_id'])
    json.dump(entity_label_to_objectId, open('/raid/liuxiao/nell_data/enwiki/entity_label_to_objectid.json', 'w'),
              ensure_ascii=False)


def add_desc_to_entity():
    entity_label_to_objectId = json.load(open('/raid/liuxiao/nell_data/enwiki/entity_label_to_objectid.json'))

    for line in tqdm(open('/raid/liuxiao/data/NePtune_data/en_wiki_span_entity_merge_coref_date.jsonl')):
        doc = json.loads(line)
        try:
            description = doc['text'].split('\n\n')[1]
        except:
            continue
        oid = doc['id']

        # get oid, objectId from title
        title = doc['title'].replace(' ', '_')
        entity_objectId = entity_label_to_objectId.get(title)

        if entity_objectId is None:
            continue
        entity = WikipediaEntity.objects.get(id=entity_objectId)
        entity.description = description
        entity.sourceId = oid

        entity.save()


if __name__ == '__main__':
    add_desc_to_entity()
