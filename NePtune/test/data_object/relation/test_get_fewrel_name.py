import os
import json

from data_object import BaseRelation

pos_stats = json.load(open('/raid/liuxiao/nell_data/extracted_facts/neptune1.0_facts_cws_pos_stats.json'))
neg_stats = json.load(open('/raid/liuxiao/nell_data/extracted_facts/neptune1.0_facts_cws_neg_stats.json'))
fsl_train = json.load(open('/raid/liuxiao/nell_data/extracted_facts/fsl_checking_extend_66.train.json'))
fsl_test = json.load(open('/raid/liuxiao/nell_data/extracted_facts/fsl_checking_extend_66.test.json'))

res = dict()

# add neptune data
for relation in neg_stats:
    if relation in pos_stats:
        res[relation] = {"pos": pos_stats[relation], "neg": neg_stats[relation], "pos_src": ['neptune']}
    else:
        res[relation] = {"pos": 0, "neg": neg_stats[relation], "pos_src": []}
for relation in pos_stats:
    if relation in res:
        continue
    res[relation] = {"pos": pos_stats[relation], "neg": 0, "pos_src": ['neptune']}

# add fsl data
for relation in fsl_train:
    if relation not in res:
        res[relation] = {"pos": 0, "neg": 0, "pos_src": []}
    res[relation]["pos"] += (fsl_train[relation]['meta']["1"] + fsl_test[relation]['meta']["1"])
    res[relation]["pos_src"].append("fsl")

# add fewrel data
filenames = list(os.listdir('/raid/liuxiao/nell_data/fewrel/by_relation'))
pids = [filename.replace('.json', '') for filename in filenames if filename.endswith('.json')]
for pid in pids:
    relation = BaseRelation.objects.get(sourceId=pid)
    if relation.text not in res:
        res[relation.text] = {"pos": 0, "neg": 0, "pos_src": []}
    res[relation.text]['pos'] += 700
    res[relation.text]['pos_src'].append('fewrel')

res = sorted(res.items(), key=lambda x: x[1]['pos'] + x[1]['neg'], reverse=True)
_res = []
for r in res:
    if r[1]['pos'] >= 25 and r[1]['neg'] >= 25:
        _res.append(r)
json.dump(_res, open('/raid/liuxiao/nell_data/extracted_facts/neptune1.0_cws_stats.json', 'w'), indent=4)
