from data_object import TripleFact
from tqdm import tqdm

for fact in tqdm(TripleFact.objects.no_cache()):
    fact.verification['is_from_abstract'] = True
    fact.save()
