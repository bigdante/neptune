from data_object import *

from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List

import json
import sys

sys.path.append('/raid/liuxiao/nell_data/fast-coref/src')


def get_tokenized_doc_char_doc(file_id):
    import spacy
    basic_tokenizer = spacy.load("en_core_web_sm")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path='/raid/liuxiao/nell_data/fast-coref/longformer_coreference_joint')

    from inference.tokenize_doc import basic_tokenize_doc, tokenize_and_segment_doc

    out = open(f'/raid/liuxiao/data/NePtune_data/coref/24_split_out/{file_id}.jsonl', 'w')
    for line in tqdm(open(f'/raid/liuxiao/data/NePtune_data/coref/24_split/{file_id}.jsonl')):
        doc = json.loads(line)
        document = doc['text']
        basic_tokenized_doc, basic_tokenized_char_doc = basic_tokenize_doc(document, basic_tokenizer)
        tokenized_doc = tokenize_and_segment_doc(
            basic_tokenized_doc, tokenizer)
        out.write(json.dumps(
            {'id': doc['id'], 'subtoken_map': tokenized_doc['subtoken_map'], 'char_list': basic_tokenized_char_doc},
            ensure_ascii=False) + '\n')
    # json.dump(data, open('/raid/liuxiao/data/NePtune_data/coref/doc_subtokenmap_charlist.json', 'w'),
    #           ensure_ascii=False)


def _mid(span: List[int]) -> float:
    return (span[0] + span[1]) / 2


def is_in_span(input_span, target_span):  # target_span should be larger
    return target_span[0] <= input_span[0] and target_span[1] >= input_span[1]


def is_right_of_span(input_span, target_span):
    return target_span[1] <= input_span[0]


def is_left_of_span(input_span, target_span):
    return target_span[0] >= input_span[1]


class SpanManager:
    def __init__(self, mentions: List[BaseMention], title_len):
        self.max_len = title_len
        self.mentions = [m for m in mentions if not m.mentionAnnotator.startswith('longformer')]
        self.coref_mentions = []

    def finalize(self, mentions):
        for mention in mentions:
            mention.temp.pop('sentenceId', None)
        return sorted(mentions + self.coref_mentions, key=lambda x: _mid(x.charSpan))

    def _upper_bound(self, tgt_span: List[int]):
        left, right = 0, len(self.mentions) - 1
        while left < right:
            mid = left + (right - left) // 2
            if _mid(self.mentions[mid].charSpan) >= _mid(tgt_span):
                if mid == 0 or _mid(self.mentions[mid - 1].charSpan) < _mid(tgt_span):
                    return mid
                else:
                    right = mid
            else:
                left = mid + 1
        return len(self.mentions)

    def get_overlapped_indices(self, tgt_span):
        index = self._upper_bound(tgt_span)
        if index == len(self.mentions):
            return []

        has_left, has_right = True, True
        res = []
        for i in range(10):
            if has_right:
                if index + i >= len(self.mentions) or is_right_of_span(self.mentions[index + i].charSpan, tgt_span):
                    has_right = False
                else:
                    # if not self.mentions[index + i].mentionAnnotator.startswith('wikipedia'):
                    res.append(index + i)

            if has_left:
                if index - i == 0 or is_left_of_span(self.mentions[index - i].charSpan, tgt_span):
                    has_left = False
                else:
                    # if not self.mentions[index - i].mentionAnnotator.startswith('wikipedia'):
                    res.append(index - i)

            if not has_right and not has_left:
                break
        return sorted(list(set(res)))

    @staticmethod
    def _overlap(tgt_span: List[int], mention: BaseMention):
        return tgt_span[0] < mention.charSpan[1] and tgt_span[1] > mention.charSpan[0]

    def process(self, cluster: List[List[int]], document):
        # get first referred entity
        referred_mention = None
        for tgt_span in cluster:
            # if tgt_span[1] - tgt_span[0] > self.max_len:  # TODO: seems to omit a lot of other entities?
            #     continue
            index = self._upper_bound(tgt_span)
            if index < len(self.mentions):
                if self._overlap(tgt_span, self.mentions[index]) and self.mentions[index].entity is not None:
                    referred_mention = self.mentions[index]
                    break
            if index > 0:
                index -= 1
                if self._overlap(tgt_span, self.mentions[index]) and self.mentions[index].entity is not None:
                    referred_mention = self.mentions[index]
                    break

        if referred_mention is None:
            return

        # create CorefMention
        for tgt_span in cluster:
            if tgt_span[1] - tgt_span[0] > self.max_len:
                continue

            # remove overlapped flair spans
            overlapped_with_anchor = False
            overlapped_indices = self.get_overlapped_indices(tgt_span)
            for i, mention_idx in enumerate(overlapped_indices):
                if self.mentions[mention_idx].mentionAnnotator.startswith('wikipedia'):
                    overlapped_with_anchor = True
                    break
            if overlapped_with_anchor:
                continue

            self.coref_mentions.append(
                BaseMention.create_coref(base_mention=referred_mention,
                                         text=document[tgt_span[0]: tgt_span[1]],
                                         span=tgt_span))
        return


def add_coref(input_file='/raid/liuxiao/data/NePtune_data/coref/final_out/test.jsonl'):
    title2objectId = json.load(open('/home/liuxiao/nell_data/enwiki/entity_label_to_objectid.json'))
    for idx, line in tqdm(enumerate(open(input_file))):
        # if idx <= 3460374:
        #     continue
        doc = json.loads(line)
        try:
            page = WikipediaPage.objects.get(source_id=doc['id'])
        except:
            continue
        if page is None:
            continue

        # 1. set correct linking of the title entity
        title_paragraph = page.paragraphs[0]
        title_sentence = title_paragraph.sentences[0]
        entity_objectId = title2objectId.get(title_sentence.text.replace(' ', '_'))
        if entity_objectId is None:
            continue
        entity = WikipediaEntity.objects.get(id=entity_objectId)

        title_sentence.mentions = [BaseMention(**{
            'text': title_sentence.text, 'charSpan': title_sentence.charSpan, 'mentionAnnotator': 'wikipedia-title',
            'mentionConfidence': 1.0, 'entity': entity
        })]
        title_sentence.save()

        # 2. acquire all mentions in page
        all_mentions = []
        paragraph_text = []
        sent_objs = []
        for paragraph_ref in page.paragraphs:
            paragraph = paragraph_ref
            paragraph_text.append(paragraph.text)
            for sentence in paragraph.sentences:
                sent_objs.append(sentence)
                for mention in sentence.mentions:
                    if mention.mentionAnnotator.startswith('longformer'):
                        continue
                    mention.temp['sentenceId'] = str(sentence.id)
                    all_mentions.append(mention)
        manager = SpanManager(all_mentions, title_len=len(title_sentence.text))
        document = "\n\n".join(paragraph_text)

        for cluster in doc['cluster']:
            manager.process(cluster, document)

        # 3. add coref mentions to sentences
        all_mentions = manager.finalize(all_mentions)
        pointer = 0
        for sentence in sent_objs[1:]:
            this_mentions = []

            while pointer < len(all_mentions):
                mention = all_mentions[pointer]

                if is_right_of_span(mention.charSpan, sentence.charSpan):
                    break
                else:
                    if is_in_span(mention.charSpan, sentence.charSpan):
                        this_mentions.append(mention)
                    pointer += 1

            sentence.mentions = this_mentions
            sentence.save()


if __name__ == '__main__':
    print(sys.argv)
    # get_tokenized_doc_char_doc(sys.argv[1])
    if len(sys.argv) > 1:
        add_coref(f'/home/liuxiao/nell_data/coref/splits/{sys.argv[1]}.jsonl')
    else:
        add_coref()
