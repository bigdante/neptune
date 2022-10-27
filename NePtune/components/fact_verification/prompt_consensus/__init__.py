from typing import List, Union
from data_object import TripleFact

import re

from collections import defaultdict
from fuzzywuzzy import fuzz
from tqdm import tqdm
import json


# Utils: Levinstein similarity
def link_list(query, values):
    if type(values) is not list:
        return fuzz.token_sort_ratio(query, values)
    result = []
    for idx, v in enumerate(values):
        result.append((idx, v, fuzz.token_sort_ratio(query, v)))
    return sorted(result, key=lambda x: x[2], reverse=True)


# General LAMBDA expressions
# True: means it satisfies the function name

def includes(fact: Union[TripleFact, str], conditions: List[str], is_lower=True):
    text = fact if type(fact) is str else fact.evidenceText
    if is_lower:
        text = text.lower()
    for c in conditions:
        if c in text:
            return True
    return False


def excludes(fact: Union[TripleFact, str], conditions: List[str], is_lower=True):
    text = fact if type(fact) is str else fact.evidenceText
    if is_lower:
        text = text.lower()
    for c in conditions:
        if c in text:
            return False
    return True


def equals(fact: Union[TripleFact, str], conditions: List[str], is_lower=True):
    text = fact if type(fact) is str else fact.evidenceText
    if is_lower:
        text = text.lower()
    if text in conditions:
        return True
    return False


def is_similar(str1, str2):
    if link_list(str1, str2) >= 70:
        return True
    return False


# Special function for certain relation

def award_received(fact: TripleFact):
    for tail_mention in re.finditer(re.escape(fact.tail), fact.evidenceText):
        prefixes = fact.evidenceText[:tail_mention.span()[0]].strip().split(' ')
        if 'named' in prefixes[:-5]:
            return False
    return True


LAMBDA_OPERATORS = {
    "name_after": lambda x: includes(x, ['named']),
    "headquarters location": lambda x: excludes(x, ['capital', 'seat']),
    "place of death": lambda x: includes(x, ['died', 'death']) if includes(x, ['burial', 'buried']) else True,
    "capital": lambda x: includes(x, ['capital', 'seat']),
    "award received": lambda x: award_received(x),
    "author": lambda x: not equals(x.head, ['he', 'his']) and not is_similar(x.head, x.tail),
    "occupant": lambda x: False if includes(x.head, ['Stadium', 'Coliseum', 'Pavilion', 'Park'],
                                            is_lower=False) and excludes(x, ['home']) else True,
    "composer": lambda x: includes(x, ['wrote', 'writ', 'compos', 'author']),
    "capital of": lambda x: includes(x, ['capital', 'seat']),
    "founded by": lambda x: includes(x, ["found", "estab", "start"]),
    "home venue": lambda x: includes(x, ['home', 'ground', 'base', 'move']),
    "lyrics by": lambda x: includes(x, ["lyric"]),
    "tributary": lambda x: not is_similar(x.head, x.tail)
}

MONGO_OPERATORS = {
    "located in the administrative territorial entity": {"$gt": 7},
    "given name": None,
    "member of sports team": {"$gt": 9},
    "family name": None,
    "occupation": None,
    "country": {"$gt": 2},
    "position held": None,
    "country of citizenship": {"$gt": 4},
    "genre": {"$gt": 5},
    "languages spoken, written or signed": {"$gt": 6},
    "performer": {"$gt": 6},
    "location": {"$gt": 8},
    "educated at": {"$gt": 6},
    "named after": {"$gt": 6},
    "place of birth": {"$gt": 7},
    "shares border with": {"$gt": 4},
    "headquarters location": {"$gt": 7},
    "sport": {"$gt": 3},
    "cast member": {"$gt": 6},
    "employer": {"$gt": 7},  # TODO: 6其实结果也挺好的，但是有一些very unexpected answers使得我不得不选择7。。。
    "father": {"$gt": 7},  # TODO: 其实结果不太妙。。。存在严重的head/tail颠倒的问题
    "spouse": {"$gt": 7},
    "has part": None,
    "place of death": {"$gt": 5},
    "owned by": {"$gt": 7},
    "capital": {"$gt": 4},
    "member of": {"$gt": 5},
    "participant in": None,
    "parent organization": {"$gt": 6},
    "applies to jurisdiction": {"$gt": 4},
    "operator": {"$gt": 6},
    "country of origin": {"$gt": 2, "$lt": 5},
    "award received": {"$gt": 7},
    "position played on team / speciality": {"$gt": 3},
    "participant": {"$gt": 4},
    "child": None,
    "director": {"$gt": 4},
    "subsidiary": None,
    "field of work": {"$gt": 6},  # TODO: 主要是tail里有挺多的人名
    "author": {"$gt": 4},
    "manufacturer": {"$gt": 3},
    "occupant": {"$gt": 5},
    "original broadcaster": {"$gt": 4},
    "instrument": {"$gt": 4},
    "work location": {"$gt": 8},
    "distributed by": None,
    "member of political party": {"$gt": 3},
    "part of the series": None,
    "military branch": {"$gt": 2},
    "publisher": {"$gt": 4},
    "present in work": None,
    "creator": {"$gt": 5},
    "form of creative work": None,
    "screenwriter": {"$gt": 5},
    "composer": {"$gt": 4},
    "capital of": {"$gt": 3},
    "military rank": {"$gt": 2},
    "conflict": {"$gt": 4},
    "league": {"$gt": 0},
    "religion": {"$gt": 1},
    "notable work": {"$gt": 2},
    "producer": {"$gt": 3},  # TODO: 存在和producation company分不清的问题；如果同一结果出现在production company中，则应移除
    "production company": {"$gt": 3},
    "founded by": {"$gt": 4},
    "replaces": {"$gt": 6},
    "product or material produced": None,
    "home venue": {"$gt": 7},  # TODO: 我发现问题可能是我用的background-context造成的
    "historic county": {"$gt": 1},
    "location of formation": {"$gt": 6},
    "place of publication": {"$gt": 8},  # TODO: 问题主要在于，混杂了很多non-geographical entity；没想好怎么解决，可能得靠p-tuning了
    "located in or next to body of water": None,
    "language of work or name": {"$gt": 3},
    "sibling": {"$gt": 5},  # TODO: 但其实正确率比较令人担忧；建议用p-tuning看一下
    "winner": {"$gt": 9},
    "narrative location": None,
    "mother": {"$gt": 8},
    "participating team": {"$gt": 1},
    "affiliation": None,
    "lyrics by": {"$gt": 3},
    "parent taxon": None,
    "record label": {"$gt": 1},
    "ethnic group": None,
    "presenter": {"$gt": 4},
    "replaced by": {"$gt": 6},
    "sports discipline competed in": {"$gt": 5},
    "residence": {"$gt": 6},
    "characters": None,
    "language used": {"$gt": 0},
    "place of burial": {"$gt": 8},
    "uses": None,
    "currency": None,
    "conferred by": {"$gt": 7},
    "coach of sports team": {"$gt": 7},
    "based on": {"$gt": 7},
    "developer": {"$gt": 0},
    "successful candidate": None,
    "operating area": {"$gt": 8},
    "country for sport": {"$gt": 4},
    "platform": {"$gt": 2},
    "candidate": {"$gt": 3},
    "noble title": {"$gt": 0},
    "tributary": {"$gt": 3},  # TODO: 过滤一下head/tail重合的
    "mouth of the watercourse": None,
    "chairperson": {"$gt": 4},
    "place served by transport hub": {"$gt": 3},
    "derivative work": None,
    "licensed to broadcast to": {"$gt": 1},
    "organizer": {"$gt": 1},
    "architect": {"$gt": 1},
    "industry": {"$gt": 0},
    "country of registry": {"$gt": 1},
    "sports season of league or competition": {"$gt": 2},
    "cause of death": {"$gt": 3},
    "connects with": {"$gt": 4},  # TODO: relation要求head/tail要为同类项，我没有想到什么好方法辨别
    "writing language": {"$gt": 5},
    "political ideology": {"$gt": 3},
    "original language of film or TV show": {"$gt": 2},  # TODO: 感觉找出来的都是country，不是language。。。
    "competition class": {"$gt": 1},
    "partner in business or sport": {"$gt": 3}
}


def check_single_fact(fact: TripleFact):
    relation = fact.relationLabel
    if MONGO_OPERATORS.get(relation):
        if '$gt' in MONGO_OPERATORS[relation] and fact.numOfAnnotators <= MONGO_OPERATORS[relation]['$gt']:
            return False
        if '$lt' in MONGO_OPERATORS[relation] and fact.numOfAnnotators >= MONGO_OPERATORS[relation]['$lt']:
            return False
    else:
        return False
    if LAMBDA_OPERATORS.get(relation) and not LAMBDA_OPERATORS[relation](fact):
        return False
    return True


def get_high_quality_fact_number():
    high_quality_fact = 0
    for _fact in tqdm(TripleFact.objects.no_cache()):
        if check_single_fact(_fact):
            high_quality_fact += 1
    print("High-quality facts:", high_quality_fact)


no_constraints = {
    "given name",
    "family name",
    "occupation",
    "position held",
    "has part",
    "participant in",
    "child",
    "subsidiary",
    "distributed by",
    "part of the series",
    "present in work",
    "form of creative work",
    "product or material produced",
    "located in or next to body of water",
    "narrative location",
    "affiliation",
    "parent taxon",
    "ethnic group",
    "characters",
    "uses",
    "currency",
    "successful candidate",
    "mouth of the watercourse",
    "derivative work"
}


def get_no_constraint_fact_number():
    no_constraint_facts = 0
    for _fact in tqdm(TripleFact.objects.no_cache()):
        if _fact.relationLabel in no_constraints:
            no_constraint_facts += 1
    print("No-constraint facts:", no_constraint_facts)


def sample_certain_relation(relation):
    hq_fact, lq_fact = list(), defaultdict(list)
    for _fact in tqdm(TripleFact.objects.no_cache()):
        # if _fact.relationLabel != relation:
        #     continue
        if check_single_fact(_fact):
            # hq_fact.append(_fact.to_json())
            pass
        else:
            lq_fact[_fact.relationLabel].append(
                {'premise': _fact.evidenceText,
                 'hypothesis': " ".join([_fact.head, _fact.relationLabel, _fact.tail]),
                 'relation': _fact.relationLabel.replace('/', '').replace(' ', '_')
                 })
    for relation in lq_fact:
        path = relation.replace('/', '').replace(' ', '_')
        json.dump(lq_fact[relation], open(f'/raid/liuxiao/nell_data/extracted_facts/per_relation/{path}.json', 'w'),
                  ensure_ascii=False)


if __name__ == '__main__':
    sample_certain_relation('work location')
