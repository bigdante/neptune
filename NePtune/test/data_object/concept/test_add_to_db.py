from data_object import BaseConcept

import json

from os.path import join
from collections import defaultdict

BASE_DIR = '/raid/liuxiao/nell_data/hailong/Schema'

# load father concept
hypernyms = defaultdict(list)
for line in open(join(BASE_DIR, 'cls-father.txt')):
    line = line.strip('\n').split('\t\t')
    hypernyms[line[0]].append(line[1])

hyponyms = defaultdict(list)
for concept, hyper_concepts in hypernyms.items():
    for hyper_con in hyper_concepts:
        hyponyms[hyper_con].append(concept)

# load head constraints
head_constraints = dict()
for line in open(join(BASE_DIR, 'cls-prop.txt')):
    line = line.strip('\n').split('\t\t')
    properties = line[1].split(';')
    head_constraints[line[0]] = properties

concept_list = set(list(hypernyms.keys())).union(hyponyms.keys())

pid2objectId = json.load(open(join(BASE_DIR, 'pid2objectId.json')))
con2objectId = dict()
for concept in concept_list:
    if concept not in head_constraints:
        head_cons = []
    else:
        head_cons = head_constraints[concept]
    doc = {"text": concept, "asHeadConstraint": [pid2objectId[pid] for pid in head_cons if pid in pid2objectId],
           "source": "Hailong-400"}
    concept_obj = BaseConcept(**doc)
    concept_obj.save()
    con2objectId[concept] = str(concept_obj.id)

json.dump(con2objectId, open(join(BASE_DIR, 'concept2objectId.json'), 'w'), indent=4)
