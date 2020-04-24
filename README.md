```python 
from bitermizer import GibbsSampler
from bitermizer import BiTermizer

b = BiTermizer(max_features=2000, ngram_range=(1,2))
b.fit_fransform(texts)


btm = GibbsSampler(num_topics=20, V=b.vocab)
topics = btm.fit_transform(b.get_biterms(), iterations=3)

btm.summarize(b.vectors,10)

btm.transform(b.unseen_biterms([lemmatize("назначить испытательный срок")])).argmax()
```
