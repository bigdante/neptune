from data_object import TripleFact

import json
import random

from tqdm import tqdm

data = []
for fact in tqdm(TripleFact.objects.no_cache()):
    data.append(json.loads(fact.to_json()))
random.shuffle(data)

json.dump(data, open('/raid/liuxiao/nell_data/extracted_facts/2022-04-21/neptune1.0_facts.json', 'w'), indent=4,
          ensure_ascii=False)
