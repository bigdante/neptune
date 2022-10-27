from data_object import WikipediaPage

from tqdm import tqdm

for page in tqdm(WikipediaPage.objects.no_cache()):
    pass
