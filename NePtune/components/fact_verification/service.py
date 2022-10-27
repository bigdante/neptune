import json
from flask import Flask, request, jsonify

from .mixed_nli import MixedNLI

processor = MixedNLI("cuda:7")


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        return json.JSONEncoder.default(self, o)


encoder = JSONEncoder()


def output_process(result):
    result = json.loads(encoder.encode(result))
    return jsonify(result)


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/query', methods=['POST'])
def index():
    """
    data here should be in the form of a list [query1, ...]
    """
    queries = request.get_json()
    return output_process(processor(queries[0], queries[1]).tolist())


app.run(host="0.0.0.0", port=21504)
