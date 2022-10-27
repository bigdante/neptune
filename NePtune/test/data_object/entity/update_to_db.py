from data_object import WikipediaEntity, WikidataEntity, WikipediaPage
from tqdm import tqdm

import json
import mongoengine.base.datastructures


# 106

def upload_missing_wikipedia_entity():
    extended_underlined_title_to_objectId = dict()
    for line in tqdm(open('/home/liuxiao/nell_data/hailong/wikipedia_wikidata_aligned_contents.jsonl')):
        doc = json.loads(line)
        try:
            entity = WikipediaEntity.objects.get(sourceId=doc['wikipedia_id'])
            continue
        except:
            try:
                wikidata_entity = WikidataEntity.objects.get(sourceId=doc['wikidata_id'])
            except:
                continue
            entity = {'text': doc['underlined_title'], 'source': 'wikipedia', 'sourceId': doc['wikipedia_id'],
                      'equivalents': {'wikidata': wikidata_entity}}
            try:
                description_page = WikipediaPage.objects.get(source_id=doc['wikipedia_page'])
                entity['describedPages'] = {'wikipedia': description_page}
                entity['description'] = ' '.join(
                    [sentence.text for sentence in description_page.paragraphs[1].sentences])
            except:
                pass
            entity_obj = WikipediaEntity(**entity)
            entity_obj.save()
            extended_underlined_title_to_objectId[doc['underlined_title']] = str(entity_obj.id)
    json.dump(extended_underlined_title_to_objectId,
              open('/raid/liuxiao/nell_data/hailong/extended_title_underline_to_objectId.json', 'w'),
              ensure_ascii=False)


def get_wikidata_entity_instance_of():
    data = dict()
    for line in tqdm(open('/home/liuxiao/LMOKG/DataPrep/kg_dump/wikidata/wikidata_latest.jsonl')):
        doc = json.loads(line)
        if 'P31' in doc['relations']:
            ins_types = [d['id'] for d in doc['relations']['P31']]
            data[doc['id']] = ins_types
    json.dump(data, open('/home/liuxiao/nell_data/hailong/wikidata_instance_of.json', 'w'))


def upload_to_wikidata_instance_of():
    wikidata_child_to_ins_type = json.load(
        open('/home/liuxiao/LMOKG/DataPrep/kg_dump/wikidata/wikidata_child_to_parents.json'))
    for entity in tqdm(WikidataEntity.objects.no_cache()):
        if entity.sourceId in wikidata_child_to_ins_type:
            ins_types = wikidata_child_to_ins_type[entity.sourceId]
            if type(entity.describedPages) is mongoengine.base.datastructures.BaseList:
                entity.describedPages = dict()
            if type(entity.types) is mongoengine.base.datastructures.BaseList:
                entity.types = dict()
            entity.types['instance_of'] = ins_types
            entity.save()


if __name__ == '__main__':
    get_wikidata_entity_instance_of()
