from data_object import WikipediaEntity

from os.path import join
from tqdm import tqdm
from urllib.parse import unquote
from collections import defaultdict

import json
import re

BASE_DIR = '/raid/liuxiao/nell_data/hailong'
concept2objectId = json.load(open(join(BASE_DIR, 'Schema', 'concept2objectId.json')))


def clean(_url):
    pat = re.compile(r'<https://en.wikipedia.org/wiki/(.*?)>')
    return unquote(pat.search(_url).group(1))


def type_list(text):
    concept_list = text.split('->')
    return [concept2objectId[c] for c in concept_list]


def main():
    # title_underline_lower_to_id = json.load(
    #     open(join('/raid/liuxiao/data/NePtune_data/titles', 'title_underline_lower_to_id.json')))
    # title_underline_to_title = json.load(
    #     open(join('/raid/liuxiao/data/NePtune_data/titles', 'title_underline_to_title.json')))
    title_underline_lower_to_objectId = dict()

    out = open(join(BASE_DIR, 'not_found_wikipedia_title.txt'), 'w')
    error_entry = 0

    for _line in tqdm(open(join(BASE_DIR, 'ins-cls.txt'))):
        line = _line.strip('\n').split('\t\t')
        url, type_text = tuple(line)

        title = clean(url)
        types = type_list(type_text)
        try:
            doc = {"text": title,
                   "url": url[1:-1],
                   "types": [types]}
            entity_obj = WikipediaEntity(**doc)
            entity_obj.save()

            title_underline_lower_to_objectId[title.lower()] = str(entity_obj.id)
        except:
            out.write(_line)
            error_entry += 1

    out.close()
    json.dump(title_underline_lower_to_objectId, open(join(BASE_DIR, 'title_underline_lower_to_objectId.json'), 'w'))
    print(len(title_underline_lower_to_objectId))
    print(error_entry)


title_underline_to_objectId = dict()
title_underline_lower_to_objectId = json.load(open(join(BASE_DIR, 'title_underline_lower_to_objectId.json')))
title_lower_cnt = defaultdict(list)
for entity in tqdm(WikipediaEntity.objects.only('text').no_cache().as_pymongo()):
    title_underline_to_objectId[entity['text']] = str(entity['_id'])
    title_lower_cnt[entity['text'].lower()].append(entity['text'])
for lower_name in list(title_lower_cnt.keys()):
    names = title_lower_cnt[lower_name]
    if len(names) > 1:
        title_lower_cnt.pop(lower_name)
json.dump(title_lower_cnt, open(join(BASE_DIR, 'title_underline_lower_duplicates.json'), 'w'), indent=4,
          ensure_ascii=False)
json.dump(title_underline_to_objectId, open(join(BASE_DIR, 'title_underline_to_objectId.json'), 'w'),
          ensure_ascii=False)
