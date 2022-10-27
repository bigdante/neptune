import os
import json

import mongoengine.base.datastructures

from data_object import BaseMention, BaseParagraph, BaseSentence, BaseRelation, WikipediaPage # , DatasetSentence, DatasetMention,

from os.path import join
from typing import Union, List
from collections import defaultdict


class PromptSchema:
    template = [
        "[ENT_DESC]\n\nSentence: [TEXT] [MENTION] [PROMPT] [MASK]",
        "\n\nExtract answers from the following sentence: [TEXT] TL;DR : [MENTION] [PROMPT] [MASK]"
        # Based on the following background: [ENT_DESC]
    ]
    STOP_RELATION = [
        'follows', 'followed by', 'contains administrative territorial entity', 'different from', 'main subject',
        'has list', 'category of associated people', 'topic\'s main category', 'category of associated people',
        'described by source', 'language used', "topic's main template", 'language of work or name',
        'on focus list of Wikimedia project', 'languages spoken, written or signed', 'has parts of the class',
    ]
    SELF_CONTAIN_RELATION = [
        'located in administrative territorial entity', 'headquarter location', 'location of formation', 'country',
        'applies to jurisdiction'
    ]

    # data (attribute) relations
    NEEDED_DATA_RELATION = {'date of birth': 'P569', 'date of death': 'P570', 'inception': 'P571',
                            'dissolved, abolished or demolished': 'P576',
                            'publication date': 'P577', 'start time': 'P580', 'end time': 'P582',
                            'point in time': 'P585', 'unemployment rate': 'P1198',
                            'start period': 'P3451', 'end period': 'P3416', 'set in period': 'P2408'}
    KILT_RELATION = {'production company': 'P272', 'crosses': 'P177', 'from narrative universe': 'P1080',
                     'occupant': 'P466', 'UTC date of spacecraft launch': 'P619', 'drafted by': 'P647',
                     'date of official opening': 'P1619', 'military rank': 'P410', 'spouse': 'P26',
                     'mouth of the watercourse': 'P403', 'chairperson': 'P488', 'participant of': 'P1344',
                     'instrument': 'P1303', 'author': 'P50', 'connecting line': 'P81',
                     'languages spoken, written, or signed': 'P1412', 'lyrics by': 'P676', 'military branch': 'P241',
                     'member of political party': 'P102', 'headquarters location': 'P159', 'award received': 'P166',
                     'place of burial': 'P119', 'taxon rank': 'P105', 'child': 'P40', 'stock exchange': 'P414',
                     'siblings': 'P3373', 'developer': 'P178', 'sport': 'P641', 'country': 'P17',
                     'publication date': 'P577', 'screenwriter': 'P58', 'architectural style': 'P149',
                     'site of astronomical discovery': 'P65', 'replaced by': 'P1366'}
    DATA_RELATION = set(json.load(open('/raid/xll/nell_data/hailong/data_triple_relations.json'))).difference(list(
        NEEDED_DATA_RELATION.values()))
        
    KILT_RELATION_PIDS = set(list(KILT_RELATION.values()))

    # EXTRACTED_RELATIONS = json.load(open('/raid/liuxiao/NePtune1.0/data/stats/fact/relation_stats.json'))
    EXTRACTED_RELATIONS = json.load(open('/raid/xll/NePtune/data/stats/fact/relation_stats.json'))

    def __init__(self):
        # type to relation
        self.pid2prompt = self.load_prompt()
        self.ins_type2props = json.load(
            open('/raid/xll/nell_data/hailong/Schema/wikidata_ins_type_to_relations.json'))
        self.head_type2props = json.load(
            open('/raid/xll/nell_data/hailong/Schema/wikidata_head_type_to_relations.json'))

    def load_prompt(self):
        pid2prompt = defaultdict(list)
        for filename in os.listdir('/raid/xll/nell_data/fewrel/ranked_prompts'):
            pid = filename.replace('.json', '')
            data = json.load(open(join('/raid/xll/nell_data/fewrel/ranked_prompts', filename)))
            for prompt in data:
                if prompt[-1] >= 0.5:
                    pid2prompt[pid].append(prompt[0])
        return pid2prompt

    def alias_processing(self, relation: BaseRelation):
        aliases = []
        for alias in [relation.text] + relation.alias[:10]:
            aliases.append(f"{alias} :")
        return aliases

    def head_constraint_filter(self, wikipedia_entity, head_constraints):
        # fetch wikidata entity
        try:
            # if type(wikipedia_entity.equivalents) is mongoengine.base.datastructures.BaseList:
            #     wikipedia_entity.equivalents = dict()
            #     wikipedia_entity.save()
            entity = wikipedia_entity.equivalents.get('wikidata')  # WikidataEntity
        except:
            entity = None
        if entity is None:
            return list(head_constraints.values()), [False] * len(head_constraints)

        # fetch entity's types
        if type(entity.types) is mongoengine.base.datastructures.BaseList:
            entity.types = dict()
            entity.save()
        if len(entity.types) == 0:
            return list(head_constraints.values()), [False] * len(head_constraints)
        entity_constraints = entity.types.get('property-as_head')
        if entity_constraints is None:
            return list(head_constraints.values()), [False] * len(head_constraints)
        entity_constraints = set(entity_constraints.keys())

        # filtering
        filtered_constraints, status = [], []
        for relation in head_constraints.values():
            # relation's head constraints
            if type(relation.HeadConstraint) is mongoengine.base.datastructures.BaseList:
                relation.HeadConstraint = dict()
            if type(relation.TailConstraint) is mongoengine.base.datastructures.BaseList:
                relation.TailConstraint = dict()
                relation.save()
            property_constraints = relation.HeadConstraint.get('wikidata')
            # relation has constraints
            if property_constraints is not None:
                if len(entity_constraints.intersection(property_constraints)) > 0:
                    filtered_constraints.append(relation)
                    status.append(True)
                else:
                    continue
            # relation doesn't have constraints
            else:
                filtered_constraints.append(relation)
                status.append(False)
        return filtered_constraints, status

    def cut_description(self, description):
        tokens = description.split(" ")[:60]
        return " ".join(tokens)

    def check_relation_obj(self, relation):
        """Only for Case 1 now"""
        if relation.sourceId in self.KILT_RELATION_PIDS \
                or len(relation.alias) > 0 \
                and relation.text not in self.STOP_RELATION \
                and not relation.text.startswith('category') \
                and relation.text in self.EXTRACTED_RELATIONS \
                and relation.sourceId not in self.DATA_RELATION:  # TODO: avoid unverifiable data properties
            return True
        return False

    def get_head_conditional_types(self, entity):
        pids = []
        if type(entity.types) is mongoengine.base.datastructures.BaseDict and entity.types.get('property-as_head'):
            for head_qid in entity.types['property-as_head']:
                pids.extend(self.head_type2props.get(head_qid) if self.head_type2props.get(head_qid) else [])
        return pids

    def get_instance_of_conditional_types(self, entity):
        pids = []
        if type(entity.types) is mongoengine.base.datastructures.BaseDict and entity.types.get('instance_of'):
            for ins_qid in entity.types['instance_of']:
                pids.extend(
                    [x[0] for x in self.ins_type2props.get(ins_qid)[:30]] if self.ins_type2props.get(ins_qid) else [])
        return pids

    # def slot_fitting(self, doc: Union[BaseParagraph, BaseSentence, DatasetSentence],
    #                  mention: Union[BaseMention, DatasetMention], relations: List[BaseRelation]):
    #     all_queries = []
    #     for relation in relations:
    #         queries = []
    #         for prompt in self.alias_processing(relation):
    #             query = self.template[1].replace('[TEXT]', doc.text)
    #             query = query.replace('[MENTION]', mention.text)
    #             query = query.replace('[PROMPT]', prompt)
    #             if mention.entity is not None:
    #                 if mention.entity['source'] == 'wikipedia':
    #                     if mention.entity.description is not None:
    #                         query = query.replace('[ENT_DESC]', self.cut_description(mention.entity.description))
    #                 elif mention.entity['source'] == 'wikidata':
    #                     if mention.entity.description is not None:
    #                         query = query.replace('[ENT_DESC]', self.cut_description(
    #                             mention.entity.text + " is " + mention.entity.description))
    #             else:
    #                 query = query.replace('[ENT_DESC]', mention.text)
    #
    #             queries.append([prompt, query])
    #         all_queries.append({"relation": relation,
    #                             "mention": mention,
    #                             "text": doc.text,
    #                             "queries": queries,
    #                             'satisfy_type_constraint': True})
    #     return all_queries

    def __call__(self, doc: Union[BaseParagraph, BaseSentence], mention: BaseMention):
        # TODO: currently only consider the first two concepts of the first concept-sequence
        if len(mention.entity.types) == 0:
            if mention.entity.equivalents.get('wikidata') is None:
                return []
            entity = mention.entity.equivalents['wikidata']
            # 1. New case: some wikipedia entity does not have annotated types
            head_types = self.get_head_conditional_types(entity)
            ins_types = self.get_instance_of_conditional_types(entity)
            pids = set(head_types + ins_types)
            head_constraints, satisfy_type_constraints = [], []
            for pid in set(pids):
                relation = BaseRelation.objects.get(sourceId=pid)
                if self.check_relation_obj(relation):
                    head_constraints.append(relation)
                    satisfy_type_constraints.append(True)
        else:
            # 2. Old case: every wikipedia entity here has manual-annotated types TODO: skip this part this time
            # return []
            concepts = mention.entity.types[0]
            head_constraints = dict()
            for concept in concepts[:1]:
                for relation in concept.asHeadConstraint:
                    if relation.sourceId not in head_constraints \
                            and relation.text not in self.STOP_RELATION \
                            and not relation.text.startswith('category') \
                            and relation.sourceId not in self.DATA_RELATION:  # TODO: avoid unverifiable data properties
                        head_constraints[relation.sourceId] = relation
            for concept in concepts[1:]:
                for relation in concept.asHeadConstraint:
                    if relation.sourceId in self.KILT_RELATION_PIDS:  # TODO: set for kilt zero_rc dataset (trex not yet)
                        head_constraints[relation.sourceId] = relation
            head_constraints, satisfy_type_constraints = self.head_constraint_filter(mention.entity, head_constraints)
        # head_constraints, satisfy_type_constraints = list(head_constraints.values()), [None] * len(head_constraints)
        all_queries = []
        for relation, is_satisfied in zip(head_constraints, satisfy_type_constraints):
            queries = []
            for prompt in self.alias_processing(relation):  # self.pid2prompt[relation.sourceId][:10] +
                query = self.template[0].replace('[TEXT]', doc.text)
                query = query.replace('[MENTION]', mention.text)
                query = query.replace('[PROMPT]', prompt)
                if mention.entity.description is not None:
                    query = query.replace('[ENT_DESC]', self.cut_description(mention.entity.description))
                else:
                    query = query.replace('[ENT_DESC]', "")

                queries.append([prompt, query])
            all_queries.append({"relation": relation,
                                "mention": mention,
                                "text": doc.text,
                                "queries": queries,
                                "satisfy_type_constraint": is_satisfied})
        return all_queries
