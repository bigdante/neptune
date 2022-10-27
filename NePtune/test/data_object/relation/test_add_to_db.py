from data_object import BaseRelation

from os.path import join
from tqdm import tqdm

import json

BASE_DIR = '/raid/liuxiao/nell_data/hailong'


def clean(text):
    text = text[1:-4]
    return text


data = dict()
for line in open(join(BASE_DIR, 'Schema', 'prop-label-alias.txt')):
    line = line.strip('\n').split('\t\t')
    if line[1] == 'rdfs:label':
        data[line[0]] = {"sourceId": line[0], "source": "wikidata-property", "alias": [], "description": None,
                         "text": clean(line[2])}
    elif line[1] == 'schema:description':
        data[line[0]]['description'] = clean(line[2])
    elif line[1] == 'skos:altLabel':
        data[line[0]]['alias'].append(clean(line[2]))
    else:
        raise NotImplementedError(f": {line}")

pid2objid = dict()

for pid, doc in tqdm(data.items()):
    rel_obj = BaseRelation(**doc)
    rel_obj.save()
    pid2objid[pid] = str(rel_obj.id)

json.dump(pid2objid, open(join(BASE_DIR, 'Schema', 'pid2objectId.json'), 'w'), indent=4)
