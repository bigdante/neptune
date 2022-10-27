import json

from os.path import join
from tqdm import tqdm

from data_object import BaseRelation


BASE_DIR = '/raid/liuxiao/nell_data/wikidata'


name2pid = json.load(open(join(BASE_DIR, 'relation_label_to_id.json')))
type_constraints = dict()
for relation_name, doc in json.load(open(join(BASE_DIR, 'wikidata_property_constraints.json'))).items():
    if relation_name in name2pid:
        type_constraints[name2pid[relation_name]] = doc


for relation in tqdm(BaseRelation.objects.no_cache()):
    constraints = type_constraints.get(relation.sourceId)
    relation.HeadConstraint, relation.TailConstraint = dict(), dict()
    if constraints is None:
        continue
    if len(constraints['head_types']):
        relation.HeadConstraint = {'wikidata': constraints['head_types']}
    if len(constraints['tail_types']):
        relation.TailConstraint = {'wikidata': constraints['tail_types']}

    relation.save()
