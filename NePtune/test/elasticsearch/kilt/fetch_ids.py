import json
from requests import post
from tqdm import tqdm

from test.elasticsearch.kilt.pymongo_client import mongo

client = mongo()


def call_es(text):
    headers = {'Content-Type': 'application/json'}
    url = 'http://localhost:9200/wikipedia_sentence/wikipedia_sentence/_search'
    data = {
        "query": {"bool": {"should": [{"match": {"text": text}}]}}
    }
    with post(url=url, headers=headers, data=json.dumps(data, ensure_ascii=False).encode('utf8'),
              auth=("nekol", "kegGER123")) as resp:
        results = resp.json()
        return data, results['hits']['hits']


# def find_obj_id(file="/home/liuxiao/data/kilt/Zero_RE/structured_zeroshot-dev-kilt.jsonl"):
def find_obj_id(file="/home/liuxiao/data/kilt/NQ/nq-dev-kilt.jsonl"):
    out = open(file.replace('.jsonl', '_matched.jsonl'), 'w')
    for idx, line in tqdm(enumerate(open(file))):
        # if idx < 1748:
        #     continue
        doc = json.loads(line)
        for output in doc['output']:
            if 'provenance' not in output:
                continue
            for provenance in output['provenance']:
                if 'start_character' not in provenance:
                    continue
                _id = provenance['wikipedia_id']
                record = client.db.knowledgesource.find_one({'_id': _id})
                if record is None:
                    raise RuntimeError(f"Record id {_id} not found in database.")
                texts = record['text'][provenance['start_paragraph_id']: provenance['end_paragraph_id'] + 1]
                text = "".join(texts)
                # print("Text:", text)
                evidence = text[provenance['start_character']: provenance['end_character']]
                if not evidence:
                    continue

                # request elasticsearch
                # print(evidence)
                query, hits = call_es(evidence)
                provenance['match'] = hits[0] if len(hits) > 0 else None
        out.write(json.dumps(doc, ensure_ascii=False) + '\n')


if __name__ == '__main__':

    find_obj_id()
