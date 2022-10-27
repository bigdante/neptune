from data_object import TripleFact

import requests
import json


def extract_by_api(doc):
    with requests.post('http://127.0.0.1:21530/query', json=doc) as resp:
        answers = resp.json()
        return answers


def test_all_relations():#relation="educated at"):
    # data = list(TripleFact.objects(relationLabel=relation))
    data = list(TripleFact.objects())
    relation = 'ALL'
    print(f"{relation}: {len(data)}")
    premises, hypothesis = [], []

    for doc in data:
        premises.append(doc['evidenceText'])
        hypothesis.append(f"{doc['head']} ; {doc['relationLabel']} ; {doc['tail']}")

    probs = extract_by_api([premises, hypothesis])

    cnt = 0
    for idx, prob in enumerate(probs):
        if prob[0] > 0.5:
            data[idx].switch_collection('triple_fact_test')
            data[idx].save(force_insert=True)
            cnt += 1
    print(f"Cleaned {relation}: {cnt}")
    # json.dump(pos_answer, open(f'/raid/liuxiao/nell_data/hhy/simple_v1/results/{relation.replace(" ", "_")}.json', 'w'),
    #           indent=4, ensure_ascii=False)


def test_certain_relations(relation="located in the administrative territorial entity"):
    data = list(TripleFact.objects(relationLabel=relation))
    print(f"{relation}: {len(data)}")
    premises, hypothesis = [], []

    for doc in data:
        premises.append(doc['evidenceText'])
        hypothesis.append(f"{doc['head']} ; {doc['relationLabel']} ; {doc['tail']}")

    probs = extract_by_api([premises, hypothesis])

    cnt = 0
    for idx, prob in enumerate(probs):
        if prob[0] > 0.5:
            data[idx].switch_collection(f'triple_fact_{relation.replace(" ", "_")}')
            data[idx].save(force_insert=True)
            cnt += 1
    print(f"Cleaned {relation}: {cnt}")
    # json.dump(pos_answer, open(f'/raid/liuxiao/nell_data/hhy/simple_v1/results/{relation.replace(" ", "_")}.json', 'w'),
    #           indent=4, ensure_ascii=False)


if __name__ == '__main__':
    test_certain_relations()
