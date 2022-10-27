from data_object import BaseRelation

from os.path import join
from tqdm import tqdm
from collections import defaultdict

import json

BASE_DIR = '/raid/liuxiao/nell_data/hailong'

label2id = dict()
for relation in BaseRelation.objects.no_cache():
    label2id[relation.text] = relation.sourceId
json.dump(label2id, open(join(BASE_DIR, 'relation_label_to_id.json'), 'w'), indent=4, ensure_ascii=False)
