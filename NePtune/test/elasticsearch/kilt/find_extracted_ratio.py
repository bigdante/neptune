from data_object import TripleFact, WikipediaPage
from tqdm import tqdm
from bson import ObjectId
from collections import defaultdict

import json


def find_ratio_in_kilt_zerore(path="/raid/liuxiao/kilt/Zero_RE/structured_zeroshot-dev-kilt_matched.jsonl"):
    cnt, extracted = 0, 0
    for line in tqdm(open(path)):
        doc = json.loads(line)
        has_found = False
        for output in doc['output']:
            for provenance in output['provenance']:
                _id = provenance['match']['_id']
                triples = [_ for _ in TripleFact.objects(evidence=ObjectId(_id))]
                has_found = len(triples) > 0
                if has_found:
                    break
            if has_found:
                break
        if has_found:
            extracted += 1
        # else:
        #     print(json.dumps(doc, ensure_ascii=False))
        #     break
        cnt += 1
    print(f"Matched / All: {extracted} / {cnt}, {extracted / cnt}")


def find_unextracted_pages():
    unextracted_abs = list()
    for page in tqdm(WikipediaPage.objects.no_cache()):
        sentence_id = page.paragraphs[0].sentences[0].id
        triples = [_ for _ in TripleFact.objects(evidence=ObjectId(sentence_id))]
        if len(triples) == 0:
            unextracted_abs.append(page.source_id)
    print(len(unextracted_abs))
    json.dump(unextracted_abs, open(f'/raid/liuxiao/nell_data/kilt/unextracted_abs_ids.json', 'w'))


def find_sections(path="/raid/liuxiao/kilt/Zero_RE/structured_zeroshot-dev-kilt_matched.jsonl"):
    cnt, abs_num = 0, 0
    for line in tqdm(open(path)):
        doc = json.loads(line)
        has_abstract = False
        for output in doc['output']:
            for provenance in output['provenance']:
                has_abstract = provenance['section'] == 'Section::::Abstract.'
                if has_abstract:
                    break
            if has_abstract:
                break
        if has_abstract:
            abs_num += 1
        cnt += 1
    print(f"{abs_num} / {cnt} = {abs_num / cnt}")


if __name__ == '__main__':
    find_ratio_in_kilt_zerore()
