from data_object import WikipediaPage, BaseSentence

from tqdm import tqdm

import json
import random


def dump_paragraphs():
    out = open('/raid/liuxiao/nell_data/neptune1.0_paragraphs.jsonl', 'w')
    for page in tqdm(WikipediaPage.objects.no_cache()):
        paragraphs = []
        for paragraph in page.paragraphs:
            sentences = []
            for sentence in paragraph.sentences:
                sentences.append(json.loads(sentence.to_json()))
            paragraphs.append(json.loads(paragraph.to_json()))
            paragraphs[-1]['sentences'] = sentences
        page_doc = json.loads(page.to_json())
        page_doc['paragraphs'] = paragraphs
        out.write(json.dumps(page_doc, ensure_ascii=False) + '\n')
    out.close()


def load_paragraphs():
    for idx, line in enumerate(open('/raid/liuxiao/nell_data/neptune1.0_paragraphs.jsonl')):
        if idx > 5:
            break
        doc = json.loads(line)
        for paragraph in doc['paragraphs']:
            for sentence in paragraph['sentences']:
                load_sentence_obj = BaseSentence.from_json(json.dumps(sentence))
                in_db_obj = BaseSentence.objects.get(id=load_sentence_obj.id)
                in_db_obj.mentions = load_sentence_obj.mentions
                in_db_obj.save()


if __name__ == '__main__':
    load_paragraphs()
