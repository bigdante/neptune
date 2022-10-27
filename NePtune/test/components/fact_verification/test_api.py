import requests
import json
import sys

sys.path.append('/raid/liuxiao/NePtune1.0')
from components.fact_verification.mixed_nli import extract_by_api

# def extract_by_api(doc):
#     with requests.post('http://127.0.0.1:21532/query', json=doc) as resp:
#         answers = resp.json()
#         print(answers)


if __name__ == '__main__':
    query = "She was launched on 27 June 1964 (sponsored by Miss Lynda Bird Johnson, the daughter of President of the United States Lyndon B. Johnson)"
    _doc = [
        [query],
        ["Lynda Bird Johnson", "father", "Lyndon B. Johnson"]
    ]
    print(extract_by_api(_doc))
