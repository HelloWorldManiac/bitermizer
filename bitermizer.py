from itertools import combinations, chain
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer


class BiTermizer(CountVectorizer):
    
    def __init__(self, *args, **kwargs):
        super(BiTermizer, self).__init__(*args, **kwargs)
        self.vocab = []
        self.vectors = []
        
    def fit_fransform(self, inp):
        self.vectors = self.fit_transform(inp).toarray()
        self.vocab = np.array(self.get_feature_names())
        
    
    def get_biterms(self):
        COMBOS = []
        for v in self.vectors:
            combo = [c for c in combinations(np.nonzero(v)[0], 2)]
            COMBOS.append(combo)
        return COMBOS
    
    def unseen_biterms(self, inp):
        COMBOS = []
        unseen_vectors = self.transform(inp).toarray()
        for v in unseen_vectors:
            combo = [c for c in combinations(np.nonzero(v)[0], 2)]
            COMBOS.append(combo)
        return COMBOS
    
