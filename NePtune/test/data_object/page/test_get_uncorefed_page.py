from data_object import *
from tqdm import tqdm

import json


def get_pages_without_coref():
    page_source_id = dict()
    for page in tqdm(WikipediaPage.objects.no_cache()):
        has_coref = False
        for paragraph in page.paragraphs:
            for sentence in paragraph.sentences:
                for mention in sentence.mentions:
                    if mention.mentionAnnotator == 'longformer-coref-joint':
                        has_coref = True
                        break
                if has_coref:
                    break
            if has_coref:
                break
        if has_coref:
            continue
        page_source_id[page.source_id] = str(page.id)
    print("Uncoref page num:", len(page_source_id))
    json.dump(page_source_id, open('/raid/liuxiao/data/NePtune_data/coref/uncoref_pages.json', 'w'))


if __name__ == '__main__':
    get_pages_without_coref()
