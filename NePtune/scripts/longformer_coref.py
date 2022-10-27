import json
import sys
import os

from tqdm import tqdm

sys.path.append('/raid/liuxiao/nell_data/fast-coref/src')


def get_coref(file_id=0, device=0, directory=1):
    from inference.model_inference import Inference
    os.makedirs(f'/raid/liuxiao/data/NePtune_data/coref/out/{directory}', exist_ok=True)
    out = open(f'/raid/liuxiao/data/NePtune_data/coref/out/{directory}/{file_id}.jsonl', 'w')
    inference_model = Inference("/raid/liuxiao/nell_data/fast-coref/longformer_coreference_joint",
                                encoder_name="/raid/liuxiao/nell_data/fast-coref/longformer_coreference_joint",
                                device=f"cuda:0")
    inference_model = inference_model
    for line in tqdm(open(f'/raid/liuxiao/data/NePtune_data/coref/split/{directory}/{file_id}.jsonl')):
        # tqdm(open(f'/home/liuxiao/NePtuen1.0/data/raw/{device}.jsonl')):
        doc = json.loads(line)
        text = doc['text']
        output = inference_model.perform_coreference(text)
        out.write(json.dumps({
            "id": doc['id'],
            "cluster": [cluster for cluster in output["clusters"] if len(cluster) > 1]
        }, ensure_ascii=False) + '\n')
        out.flush()


def split_data(interval=1007916):
    out = None
    for idx, line in tqdm(
            enumerate(open('/raid/liuxiao/data/NePtune_data/en_wiki_span_entity_merge_coref_date_shuf.jsonl'))):
        if idx % interval == 0:
            out = open(f'/raid/liuxiao/data/NePtune_data/coref/split/{idx // interval}.jsonl', 'w')
        out.write(line)


def transform_token_to_char():
    def flatten(iterable):
        res = []
        for item in iterable:
            for i in item:
                res.append(i)
        return res

    # data = json.load(open('/raid/liuxiao/data/NePtune_data/coref/out/sample.jsonl'))
    # data[data['id']] = {'cluster': data['cluster']}
    # text = json.load(open('/raid/liuxiao/data/NePtune_data/sample.json'))['text']
    data = json.load(open('/raid/liuxiao/data/NePtune_data/coref/out/token_span_coref.json'))

    out = open('/raid/liuxiao/data/NePtune_data/coref/final_out/char_span_coref.jsonl', 'w')

    for file_id in range(24):
        print(file_id)
        for line in tqdm(open(f'/raid/liuxiao/data/NePtune_data/coref/24_split_out/{file_id}.jsonl')):
            doc = json.loads(line)
            token_res = data.get(doc['id'])
            if token_res is None:
                continue
            clusters = token_res

            subtoken_map, char_list = doc['subtoken_map'], doc['char_list']
            char_list = flatten(char_list)

            new_clusters = []

            for cluster in clusters:
                new_clusters.append([])
                for mention in cluster:
                    subtoken_span, mention = tuple(mention)
                    token_span = [subtoken_map[subtoken_span[0]], subtoken_map[subtoken_span[1]]]
                    char_span = [char_list[token_span[0]][0], char_list[token_span[1]][1]]
                    new_clusters[-1].append(char_span)

                    # assert text[char_span[0]: char_span[1]] == mention

            out.write(json.dumps({'id': doc['id'], 'cluster': new_clusters}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # print(sys.argv)
    # get_coref(file_id=int(sys.argv[1]), device=0, directory=int(sys.argv[2]))
    transform_token_to_char()
