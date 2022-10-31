from components import PromptSchema, PromptExtractor, MixedNLIWrapper
from data_object import BaseSentence, BasePage, BaseParagraph, BaseMention, TripleFact, WikipediaEntity, WikipediaPage
from controller.offsets import OFFSETS, PORT_MAPPING, FV_MAPPING

from tqdm import tqdm
from datetime import datetime
from os.path import join
from functools import partial

import json
import os
import sys


def get_extract_port(process_id):
    return PORT_MAPPING[process_id]


class ControllerV1:
    def __init__(self, process_id=0, offset=0, device=0):
        # TODO:模块作用，返回sentence和对应的结合了prompt的queries
        self.relation_schema = PromptSchema()
        # TODO:模块作用，返回输入的queries查找到的结果
        self.fact_extraction = PromptExtractor(port=get_extract_port(process_id))

        # 这里传给fact_verification一个doc，则会经过request.post请求将answer返回
        # TODO:模块作用，返回结果的分数值
        self.fact_verification = partial(MixedNLIWrapper,
                                         args=(process_id, FV_MAPPING[process_id]))  # MixedNLI(f'cuda:{device}')

        self.process_id = process_id
        # TODO：offset是用于做什么？是为了多卡训练时候，每个卡负责一段的数据吗
        self.offset = offset

        # self.true_offset = int(sys.argv[4]) + self.process_id * 128671 + self.offset
        # TODO：true_offset的作用
        self.true_offset = OFFSETS[self.process_id]
        # TODO：true_end的作用，这个数字的意义
        self.true_end = min(self.true_offset + 30952, 6047494)
        # TODO：sentence_offset的作用
        self.sentence_offset = -1

        print(f"\n#######################\nTrue offset: {self.true_offset}\n#######################")

        # TODO：这个对应entity表中的text和id
        self.underline_title_to_objectId = json.load(
            open('/raid/xll/nell_data/hailong/title_underline_to_objectId.json'))

        # load relation filter
        # 这个记录relation的出现次数
        self.relation_freq = json.load(
            open(join('/raid/xll/nell_data/wikidata/wikidata_relation_tail_uniqueness_frequency_stats.json')))
        # relation以及对应的别名
        self.relation_tails = json.load(
            open(join('/raid/xll/nell_data/wikidata/wikidata_relation_to_candidate_aliases.json')))

        # TODO：threshold的作用
        self.threshold = 4.0

        # create negative logging
        self.log_dir = f'/raid/xll/nell_code/log/{datetime.now()}'
        self.log_iter_dir = f'/raid/xll/nell_code/log/iterations'
        os.makedirs(self.log_dir)
        os.makedirs(self.log_iter_dir, exist_ok=True)
        self.log = open(join(self.log_dir, 'invalid_facts.jsonl'), 'w')
        self.log_process = open(join(self.log_dir, 'processed_sentences.jsonl'), 'w')

        if os.path.exists(join(self.log_iter_dir, f'{process_id}')):
            self.offset = int(open(join(self.log_iter_dir, f'{process_id}')).read().strip())

    # 由于有的mention没有对应的entity，就需要通过underline_title_to_objectId表 + entity表进行查找
    def map_to_entity(self, mention: BaseMention):
        if mention.temp.get('entry'):
            objectId = self.underline_title_to_objectId.get(mention.temp['entry'].replace(' ', '_'))
            if objectId is not None:
                return WikipediaEntity.objects.get(id=objectId)

    '''
        从（Wikipeida）Page数据表中获取到区间的所有句子
    '''

    def yield_sentence_from_page(self):
        while self.true_offset < self.true_end:
            pages = WikipediaPage.objects[self.true_offset: min(self.true_offset + 100, self.true_end)]
            for page in pages:
                for paragraph in page.paragraphs:
                    for sentence in paragraph.sentences:
                        yield sentence
            # TODO: 为什么要100
            self.true_offset += 100

    '''
        这个模块没有使用。
        从数据库的sentence表中得到所有区间范围的句子
    '''

    def yield_sentence(self):
        while self.true_offset < self.true_end:
            sentences = BaseSentence.objects[self.true_offset: min(self.true_offset + 100, self.true_end)]
            for s in sentences:
                yield s
            self.true_offset += 100

    '''
        从page中得到所有的sentence
    '''

    def get_sentence(self):
        for idx, sentence in tqdm(enumerate(self.yield_sentence_from_page())):
            self.sentence_offset += 1

            if self.sentence_offset < self.offset:
                continue
            # TODO：为了记录啥？？每是个句子记录一次？
            if idx % 10 == 0:
                with open(join(join(self.log_iter_dir, f'{self.process_id}')), 'w') as f:
                    f.write(str(idx))
            yield sentence

    '''
        获取句子中的mention，也就是句子中的head entity。
        
    '''

    def get_valid_mention(self, sentence: BaseSentence):
        for mention in sentence.mentions:
            # 如果没有entity（对应wikipedia类，entity表），说明这个mention还没有和entity对应，则需要通过temp.entry找到entity
            if mention.entity is None:
                mention.entity = self.map_to_entity(mention)
                if mention.entity is None:
                    continue
                sentence.save()
            # TODO:应该是不为空时候save，这里似乎有点影响效率，所以调整到上面
            # sentence.save()
            # mention.entity.fetch()
            # mention.entity = WikipediaEntity.objects.get(id=mention.entity)
            yield mention

    '''
        得到fact，
    '''

    def get_facts(self, sentence: BaseSentence, mention: BaseMention):
        # 这里的gueries得到了每个句子的输入数据，也就是prompt等信息
        queries = self.relation_schema(sentence, mention)

        return self.fact_extraction(sentence, queries)

    def get_verification(self, facts: TripleFact):
        verified_facts = []

        def generate_premise_hypothesis(triple_fact: TripleFact):
            premise = triple_fact.evidence.text
            _hypothesises = [f"{triple_fact.head} {annotator} {triple_fact.tail}" for annotator in
                             [triple_fact.relationLabel] + triple_fact.annotator]
            return [premise] * len(_hypothesises), _hypothesises

        for fact in facts:
            premises, hypothesises = generate_premise_hypothesis(fact)
            predictions = self.fact_verification((premises, hypothesises))  # .tolist()

            annotators, verification = [], dict()
            for answer, pred in zip([fact.relationLabel] + fact.annotator, predictions):
                if pred[0] > 0.5:
                    annotators.append(answer)
                    verification[answer] = dict(zip(['entail', 'neutral', 'not_entail'], pred))

            fact.verification["mixed-nli"] = verification
            fact.annotator = annotators

            # 1. facts that do not comply unique tail constraints
            pass_check = True
            if fact.relationLabel in self.relation_freq and fact.relationLabel in self.relation_tails and \
                    self.relation_freq[fact.relationLabel] >= self.threshold:
                if fact.tail in self.relation_tails[fact.relationLabel]:
                    uniqueness_pred = [1.0, 0.0, 0.0]
                else:
                    uniqueness_pred = [0.0, 0.0, 1.0]
                fact.verification['uniqueness-check'] = dict(zip(['entail', 'neutral', 'not_entail'], uniqueness_pred))
                if uniqueness_pred[0] == 0.0:
                    pass_check = False

            # 2. facts that are too naive
            if pass_check and fact.relationLabel not in self.relation_schema.SELF_CONTAIN_RELATION:
                if fact.tail == fact.head or fact.relationLabel in fact.tail:
                    fact.verification['self-contained-check'] = False
                    pass_check = False
                else:
                    fact.verification['self-contained-check'] = True
            else:
                fact.verification['self-contained-check'] = True

            if pass_check and len(fact.annotator) > 0:
                fact.numOfAnnotators = len(fact.annotator)
                fact.save()
                verified_facts.append(fact)
            else:
                # invalid facts hit here
                self.log.write(fact.to_json() + '\n')

        return verified_facts

    def run(self):
        for sentence in self.get_sentence():
            num_of_facts = 0
            # mentions 中存着该句子的所有head，就是要预测该head对应的relation和tail。
            for mention in self.get_valid_mention(sentence):
                # 这里将sentence和对应的mention输入到get_facts得到glm模型得到的所有fact，也就是预测head的tail和relation
                facts = self.get_facts(sentence, mention)
                print("facts:", facts)
                verified_facts = self.get_verification(facts)
                num_of_facts += len(verified_facts)
            # if num_of_facts > 0:
            #     sentence.temp['has_prediction'] = True
            # sentence.temp['last_processed'] = datetime.now()
            # sentence.save()
            self.log_process.write(str(sentence.id) + '\n')


if __name__ == '__main__':
    print(sys.argv)
    # controller = ControllerV1(process_id=int(sys.argv[1]))
    controller = ControllerV1(process_id=0)
    controller.run()
