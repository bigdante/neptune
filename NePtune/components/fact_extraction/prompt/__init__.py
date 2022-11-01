import requests
import json

from typing import List, Dict
from collections import defaultdict

from data_object import TripleFact


def flatten(t):
    return [item for sublist in t for item in sublist]


class PromptExtractor:
    """
    A natural-language based prompt extractor.
    """

    def __init__(self, port=21534):
        self.port = port

    def extract_by_api(self, doc):
        try:
            with requests.post(f'http://127.0.0.1:{self.port}/query', json=doc) as resp:
                answers = resp.json()
                return answers
        except:
            return [''] * len(doc)

    def get_top_answer(self, answer2queries):
        top_answer = sorted(answer2queries.items(), key=lambda x: len(x[1]), reverse=True)[0]
        return top_answer

    def __call__(self, sentence, queries: List[Dict]):
        """"
        queries structure: [{"relation": ..., "mention": BaseMention, "text": str,
        "queries": [[prompt, query], ...]}, ... ]
        """
        extracted_facts = []
        all_prompts, all_queries, query_id_map, all_answers = [], [], [], []
        for idx, query_per_rel in enumerate(queries):
            all_prompts.append([query[0] for query in query_per_rel['queries']])
            all_queries.append([query[1] for query in query_per_rel['queries']])
            query_id_map.extend([idx] * len(query_per_rel['queries']))
            all_answers.append([])

        q = flatten(all_queries)
        all_responses = self.extract_by_api(q)

        for idx, response in enumerate(all_responses):
            all_answers[query_id_map[idx]].append(response)

        for prompts, answers, query_per_rel in zip(all_prompts, all_answers, queries):
            answer2queries = defaultdict(list)
            for annotator, answer in zip(prompts, answers):
                answer2queries[answer].append(annotator)

            top_answer = self.get_top_answer(answer2queries)
            if top_answer[0] in {"", "the", "a", "an", '"'} or \
                    len(top_answer[0]) >= 50 or \
                    top_answer[0] in {query_per_rel['mention'].text}:
                continue
            fact = {"annotator": top_answer[1],
                    "head": query_per_rel['mention'].text,
                    "relationLabel": query_per_rel['relation'].text,
                    "evidenceText": sentence.text,
                    "tail": top_answer[0],
                    "headSpan": query_per_rel['mention'].charSpan,
                    "relation": query_per_rel['relation'],
                    "evidence": sentence,
                    "headWikipediaEntity": query_per_rel['mention'].entity,
                    "verification": {"wikidata-type-constraint": query_per_rel['satisfy_type_constraint']}}
            extracted_facts.append(TripleFact(**fact))

        return extracted_facts
