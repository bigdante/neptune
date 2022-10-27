import spacy
from spacy import Language


@Language.component('set_custom_boundaries')
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == ".(" or token.text == ").":
            doc[token.i + 1].is_sent_start = True
        elif token.text in ['(', '[', ']', ')', ':']:
            doc[token.i + 1].is_sent_start = False
    return doc


class SpacySentSplitter:
    def __init__(self, parameters=None):
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.add_pipe('set_custom_boundaries', before="parser")

    def predict(self, text):
        preds = []
        doc = self.nlp(text)
        for sent_idx, sent in enumerate(doc.sents):
            preds.append({
                "idx": sent_idx,
                "text": sent.text,
                "sent_span": [sent.start_char, sent.end_char]
            })
        return preds
