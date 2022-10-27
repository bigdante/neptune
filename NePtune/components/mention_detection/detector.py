from tqdm import tqdm
from datetime import datetime

import json

from components.mention_detection.flair_ner import Flair
from components.mention_detection.spacy_splitter import SpacySentSplitter
from data_object import WikipediaPage, BaseParagraph, BaseSentence, BaseMention


class Annotator(object):
    def __init__(self, docid="en_wiki_span_entity_merge_coref_date_shuf"):
        self.docid = docid
        self.ner = Flair()
        print("NER model loaded...")
        self.splitter = SpacySentSplitter()
        print("Spacy loaded...")

    def run(self):

        def is_in_span(input_span, target_span):  # target_span should be larger
            return target_span[0] <= input_span[0] and target_span[1] >= input_span[1]

        def is_right_of_span(input_span, target_span):
            return target_span[1] <= input_span[0]

        def offset(span, offset):
            return [span[0] + offset, span[1] + offset]

        def filter_duplicate(mentions):
            _mentions = []
            for idx, m_obj in enumerate(mentions[:-1]):
                if is_in_span(mentions[idx + 1].charSpan, m_obj.charSpan):
                    if 'the ' + mentions[idx + 1].text == m_obj.text or mentions[idx + 1].text == m_obj.text:
                        mentions[idx + 1].mentionType = m_obj.mentionType
                        continue
                _mentions.append(m_obj)
            if len(mentions) > 0:
                _mentions.append(mentions[-1])
            return _mentions

        for line in tqdm(open(f'/raid/liuxiao/data/NePtune_data/{self.docid}.jsonl')):
            doc = json.loads(line)

            # create Page object
            page = WikipediaPage(source_id=doc['id'])

            para_objs = []
            pointer = -2
            latest_anchor_mention_idx = 0
            existed_spans = set()
            for para_idx, para_text in enumerate(doc['text'].split('\n\n')):
                pointer += 2

                # create Paragraph object
                para = BaseParagraph(inPageId=para_idx, charSpan=[pointer, pointer + len(para_text)])

                para_mentions = []
                ner_output = self.ner.predict([para_text])
                for mention_idx, mention in enumerate(ner_output['mentions']):
                    para_mentions.append(
                        {"text": mention['text'],
                         "charSpan": [pointer + mention['start_pos'], pointer + mention['end_pos']],
                         "mentionType": mention['labels'][0].value,
                         "mentionConfidence": float(mention['labels'][0].score),
                         "mentionAnnotator": "Flair-notonotes-fast"})

                sentences = self.splitter.predict(para_text)
                sent_objs = []
                latest_ner_mention_idx = 0
                for sent in sentences:
                    sent_obj = BaseSentence(text=sent['text'], charSpan=offset(sent['sent_span'], pointer),
                                            inParaId=sent['idx'])
                    sent_mentions = []

                    # collect mentions from ner
                    while latest_ner_mention_idx < len(para_mentions) and not is_right_of_span(
                            para_mentions[latest_ner_mention_idx]['charSpan'], sent_obj.charSpan):
                        if is_in_span(para_mentions[latest_ner_mention_idx]['charSpan'], sent_obj.charSpan):
                            sent_mentions.append(BaseMention(**para_mentions[latest_ner_mention_idx]))
                        latest_ner_mention_idx += 1

                    # collect mentions from wikipedia anchor
                    while latest_anchor_mention_idx < len(doc['span_entity']):
                        m = doc['span_entity'][latest_anchor_mention_idx]

                        if is_right_of_span(m[0], sent_obj.charSpan):
                            break
                        else:
                            if m[2] == 1 and is_in_span(m[0], sent_obj.charSpan):
                                if str(m[0]) not in existed_spans:
                                    sent_mentions.append(BaseMention(text=doc['text'][m[0][0]: m[0][1]],
                                                                     charSpan=m[0],
                                                                     mentionAnnotator="wikipedia-anchor",
                                                                     mentionConfidence=1.0,
                                                                     temp={"entry": m[1]}))
                                    existed_spans.add(str(m[0]))
                            latest_anchor_mention_idx += 1
                    sent_obj.mentions = filter_duplicate(
                        sorted(sent_mentions, key=lambda x: x.charSpan[0] + x.charSpan[1]))
                    sent_obj.save()
                    sent_objs.append(sent_obj)

                para.sentences = sent_objs
                para.save()
                para_objs.append(para)

                pointer += len(para_text)

            page.paragraphs = para_objs
            page.save()


if __name__ == '__main__':
    annotator = Annotator()
    annotator.run()
