import json

from tqdm import tqdm

missing_qid = set(json.load(open('missing_qid_as_tail.json')))
out = open('missing_qid_as_tail_docs.jsonl', 'w')
for line in tqdm(open('wikidata_latest.jsonl')):
    doc = json.loads(line)
    if doc['id'] in missing_qid:
        data = {'text': doc['DisplayName_En'], 'sourceId': doc['id'], 'aliases': doc['Aliases_En'],
                'source': 'wikidata'}
        out.write(json.dumps(data, ensure_ascii=False) + '\n')
