from data_object import BaseConcept

import json

from os.path import join
from collections import defaultdict

concept2relation = dict()
for concept in BaseConcept.objects.no_cache():
    concept2relation[concept.text] = [r.text for r in concept.asHeadConstraint]
json.dump(concept2relation, open('/raid/liuxiao/nell_data/hailong/Schema/concept2head_constraints.json', 'w'), indent=4,
          ensure_ascii=False)
