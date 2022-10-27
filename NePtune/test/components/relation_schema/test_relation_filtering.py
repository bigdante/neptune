from components import PromptSchema
from data_object import WikipediaEntity, BaseSentence

import json

from tqdm import tqdm


def test_filtering():
    cnt = 0
    schema = PromptSchema()
    for idx, entity in tqdm(enumerate(WikipediaEntity.objects.no_cache())):
        concepts = entity.types[0]
        head_constraints = dict()
        for concept in concepts[:2]:
            for relation in concept.asHeadConstraint:
                if relation.sourceId not in head_constraints and relation.text not in schema.STOP_RELATION:
                    head_constraints[relation.sourceId] = relation
        _head_constraints, status = schema.head_constraint_filter(entity, head_constraints)
        if len(_head_constraints) < len(head_constraints):
            cnt += 1
            print(f"{cnt}/{idx}\t{entity.text}\t{entity.sourceId}\t{len(head_constraints)} -> {len(_head_constraints)}")


def test_certain_sentence():
    schema = PromptSchema()
    sentence = BaseSentence.objects(id='624bce12c20df149acb95411').first()
    mention = sentence.mentions[0]

    concepts = mention.entity.types[0]
    head_constraints = dict()
    for concept in concepts[:2]:
        for relation in concept.asHeadConstraint:
            if relation.sourceId not in head_constraints \
                    and relation.text not in schema.STOP_RELATION \
                    and not relation.text.startswith('category') \
                    and relation.sourceId not in schema.DATA_RELATION:  # TODO: avoid unverifiable data properties
                head_constraints[relation.sourceId] = relation
    res = schema.head_constraint_filter(mention.entity, head_constraints)
    print([r.text for r in res[0]])
    # print([r.text for r in head_constraints.values()])


if __name__ == '__main__':
    test_certain_sentence()
