from .base_fact import *

from ..mention import BaseMention


class TripleFact(BaseFact):
    """
    Store text-based triple evidence.
    """
    head = StringField(required=True)
    relationLabel = StringField(required=True)
    tail = StringField(required=True)

    headSpan = ListField(IntField(), required=True)
    relation = ReferenceField('BaseRelation', required=True)
    tailSpan = ListField(IntField())

    headWikidataEntity = ReferenceField('WikidataEntity')
    headWikipediaEntity = ReferenceField('WikipediaEntity')

    evidence = GenericReferenceField(required=True)
    evidenceText = StringField()

    verification = DictField()
    upVote = IntField()
    downVote = IntField()
    isNewFact = BooleanField()

    meta = {
        "collection": "triple_fact_missing",
        "db_alias": "NePtune"
    }
