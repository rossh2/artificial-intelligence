from itertools import product
from typing import List, Iterable

import numpy as np
from nltk.corpus import wordnet
from nltk.corpus.reader import Synset

from analogy.word_similarity import WordSimilarities


class WordnetSimilarities(WordSimilarities):
    # WordNet parts of speech
    NOUN = wordnet.NOUN
    ADJECTIVE = wordnet.ADJ

    # Similarity metrics implemented in NLTK
    PATH = 'Path'
    WUP = 'Wu-Palmer'
    LCH = 'Leacock-Chodorow'

    def __init__(self, pos: str, vocabulary: Iterable[str] = None, similarity_method=WUP,
                 max_neighbour_count: int = None, min_max_neighbour_similarity: float = None,
                 max_synsets: int = None):
        self.pos = pos
        self.similarity_method = similarity_method
        self.max_synsets = max_synsets
        super().__init__(vocabulary=vocabulary,
                         max_neighbour_count=max_neighbour_count,
                         min_max_neighbour_similarity=min_max_neighbour_similarity)

    def build_similarities(self) -> np.ndarray:
        vocab_size = len(self.vocabulary)
        similarities = np.zeros(shape=(vocab_size, vocab_size))

        for i in range(vocab_size):
            word1 = self.vocabulary[i]
            if not self.has_synset(word1):
                print(f'Warning: no synsets found for word "{word1}",', end=' ')
                if word1 == "knockoff":
                    word1 = "counterfeit"
                    print('substituting "counterfeit" as near-synonym')
                else:
                    # Skip this word
                    print('setting all similarities to 0')
                    continue
            for j in range(i, vocab_size):
                word2 = self.vocabulary[j]
                if word2 == "knockoff":
                    word2 = "counterfeit"
                    print(f'Warning: substituting "counterfeit" as near-synonym for "knockoff" '
                          f'for similarity with "{word1}"')
                wupsim = self.pairwise_word_similarity(word1, word2, pos=self.pos, max_synsets=self.max_synsets)
                similarities[i, j] = wupsim
                similarities[j, i] = wupsim

        return similarities

    def has_synset(self, word: str) -> bool:
        return not not self.get_synset(word, self.pos)

    @staticmethod
    def get_synset(word: str, pos: str) -> List[Synset]:
        if ' ' in word:
            # Noun-noun compound, take root
            print(f'Warning: splitting compound {word} by spaces and taking root (last word)')
            word = word.split(' ')[-1]

        return wordnet.synsets(word, pos)

    def pairwise_word_similarity(self, word1: str, word2: str, pos: str,
                                 max_synsets: int = None) -> float:
        synsets1 = self.get_synset(word1, pos=pos)
        synsets2 = self.get_synset(word2, pos=pos)

        if max_synsets is not None:
            synsets1 = synsets1[:max_synsets]
            synsets2 = synsets2[:max_synsets]

        synset_pairs = product(synsets1, synsets2)

        similarities = []
        for synset1, synset2 in synset_pairs:
            sim = self.synset_similarity(synset1, synset2, self.similarity_method)
            similarities.append(sim)

        if not similarities:
            print(f'Warning: could not calculate similarity for words "{word1}" and "{word2}", no synsets')

        return max(similarities) if similarities else 0

    @classmethod
    def synset_similarity(cls, synset1: Synset, synset2: Synset, similarity_method: str = WUP) -> float:
        if similarity_method == cls.WUP:
            sim = synset1.wup_similarity(synset2)
        elif similarity_method == cls.LCH:
            sim = synset1.lch_similarity(synset2)
        elif similarity_method == cls.PATH:
            sim = synset1.path_similarity(synset2)
        else:
            raise ValueError(f'Unknown similarity method {similarity_method}')
        sim = sim if sim is not None else 0
        return sim
