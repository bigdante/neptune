import requests
import json


def extract_by_api(doc):
    with requests.post('http://127.0.0.1:21540/query', json=doc) as resp:
        answers = resp.json()
        print(json.dumps(list(zip(doc, answers)), indent=4))


if __name__ == '__main__':
    desc = "Lyndon Baines Johnson (; August 27, 1908January 22, 1973), often referred to by his initials LBJ, was an American politician who served as the 36th president of the United States from 1963 to 1969. Formerly the 37th vice president from 1961 to 1963, he assumed the presidency following the assassination of President John F. Kennedy. A Democrat from Texas, Johnson also served as a United States Representative and as the Majority Leader in the United States Senate. Johnson is one of only four people who have served in all four federal elected positions."
    query = "She was launched on 27 June 1964 (sponsored by Miss Lynda Bird Johnson, the daughter of President of the United States Lyndon B. Johnson), and commissioned on 6 February 1965 with Captain William H. Shaw in command. Lyndon B. Johnson 's father is [MASK]"
    _doc = [
        f"{desc}\n\n{query}"
    ]
    extract_by_api(_doc)
