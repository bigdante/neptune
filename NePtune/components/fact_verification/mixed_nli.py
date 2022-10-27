from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import requests
import os

from typing import List
os.environ['TRANSFORMERS_CACHE'] = '/raid/liuxiao/checkpoints/cache/'

MAPPING = [
    "192.93.237.27",
    "192.168.90.247",
    "192.223.130.216",
    "192.255.250.42"
]

def get_node(process_id):
    if 0 <= process_id <= 38:
        return 0
    elif 39 <= process_id <= 77:
        return 1
    elif 78 <= process_id <= 116:
        return 2
    else:
        return 3


class MixedNLI:
    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    max_length = 256

    def __init__(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained(self.hg_model_hub_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.hg_model_hub_name).to(device)

    def __call__(self, premise: List[str], hypothesis: List[str]) -> np.array:
        with torch.no_grad():
            tokenized_input_seq_pair = self.tokenizer.batch_encode_plus(list(zip(premise, hypothesis)),
                                                                        max_length=self.max_length,
                                                                        return_token_type_ids=True, truncation=True,
                                                                        padding=True)
            input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long()
            # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
            token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long()
            attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long()

            outputs = self.model(input_ids.to(self.model.device),
                                 attention_mask=attention_mask.to(self.model.device),
                                 token_type_ids=token_type_ids.to(self.model.device),
                                 labels=None)

            predicted_probability = torch.softmax(outputs[0], dim=1).cpu().numpy()

            return predicted_probability


def extract_by_api(doc, args):
    #
    with requests.post(f'http://{MAPPING[get_node(args[0])]}:{21500 + args[1]}/query', json=doc) as resp:
        answers = resp.json()
        return answers
