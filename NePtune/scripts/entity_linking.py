import json

from tqdm import tqdm

VALUE_TYPES = ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]


def read_and_transform():
    out = open('../data/out/neptune1.0_inputs.jsonl', 'w')
    for line in tqdm(open('/home/liuxiao/NePtune1.0/data/neptune1.0_paragraphs.jsonl')):
        page = json.loads(line)
        for paragraph in page['paragraphs'][1:]:
            sentences = [sentence['text'] for sentence in paragraph['sentences']]
            for sentence in paragraph['sentences']:
                for idx, mention in enumerate(sentence['mentions']):
                    # try:
                    if mention['mentionAnnotator'] == 'wikipedia-anchor' or mention.get('mentionType') in VALUE_TYPES:
                        continue
                    # except:
                    #     print(json.dumps(mention, indent=4))
                    span = [mention['charSpan'][i] - sentence['charSpan'][i] for i in range(0, 2)]
                    out.write(json.dumps({
                        "sentence_id": sentence['_id']['$oid'],
                        "inSentId": idx,
                        "left_context": " ".join(sentences[:sentence['inParaId']] + [sentence['text'][:span[0]]]),
                        "mention": mention['text'],
                        "right_context": " ".join(sentences[sentence['inParaId'] + 1:] + [sentence['text'][span[1]:]]),
                    }, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    read_and_transform()
