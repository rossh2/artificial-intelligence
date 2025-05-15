import csv
from collections import OrderedDict
from typing import List, Iterable

import numpy as np
import pandas as pd
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

from analogy.word_embeddings import WordEmbeddings
from utils.io import read_words

GLOVE_6B_100D_PATH = 'data/glove/glove.6B.100d.txt'
GLOVE_6B_300D_PATH = 'data/glove/glove.6B.300d.txt'


class GloveEmbeddings(WordEmbeddings):
    NOUN = wordnet.NOUN

    def __init__(self, glove_path: str = GLOVE_6B_300D_PATH,
                 distance_metric=WordEmbeddings.COSINE_SIMILARITY,
                 vocabulary: Iterable[str] = None,
                 max_neighbour_count: int = None, min_max_neighbour_similarity: float = None):
        super().__init__(glove_path,
                         vocabulary=vocabulary,
                         distance_metric=distance_metric,
                         max_neighbour_count=max_neighbour_count,
                         min_max_neighbour_similarity=min_max_neighbour_similarity)
        self.lemmatizer = WordNetLemmatizer()

    def load_embeddings(self):
        return pd.read_table(self.embeddings_path, sep=' ', index_col=0, header=None, quoting=csv.QUOTE_NONE,
                             keep_default_na=False)

    def find_words_within_distance(self, word: str, distance: float, max_words: int = 100,
                                   lemmatize: bool = True, pos: str = wordnet.NOUN,
                                   wordnet_filter: bool = False) -> List[str]:
        if word not in self.embeddings.index:
            print(f'Warning: unknown word {word}, returning empty list of neighbours')
            return []

        if self.distance_graph:
            return self.get_neighbors(word, n=max_words, min_or_max_value=distance)

        vector = self.get_vector(word)
        delta = self.get_distances(vector)
        delta = np.sort(delta)  # Sort ascending so we get the closest ones when taking [:max_words]
        indices = np.where(delta < distance)
        close_words = [self.embeddings.iloc[i].name for i in indices]
        close_words = self.filter_neighbours(word, close_words, lemmatize, pos, wordnet_filter)
        close_words = close_words[:max_words]
        return close_words

    def find_nearby_words(self, word: str, neighbours_per_word: int,
                          lemmatize: bool = True, pos: str = wordnet.NOUN, wordnet_filter: bool = False,
                          exclude_secondary_lemmas: bool = False, words_to_exclude: List[str] = None,
                          max_neighbours_to_search: int = 25) -> List[str]:
        if word not in self.embeddings.index:
            print(f'Warning: unknown word {word}, returning empty list of neighbours')
            return []

        if self.distance_graph:
            return self.get_neighbors(word, neighbours_per_word)

        if neighbours_per_word > max_neighbours_to_search:
            raise ValueError(f'neighbours_per_word {neighbours_per_word} is greater than max_neighbours_to_search '
                             f'{max_neighbours_to_search}; please configure this value to be higher')

        word_vector = self.get_vector(word)
        word_neighbours = self.find_closest_words(word_vector, max_neighbours_to_search + 1)
        new_neighbours = self.filter_neighbours(word, neighbours=word_neighbours, lemmatize=lemmatize, pos=pos,
                                                wordnet_filter=wordnet_filter,
                                                exclude_secondary_synset_lemmas=exclude_secondary_lemmas,
                                                words_to_exclude=words_to_exclude)
        new_neighbours = new_neighbours[:neighbours_per_word]
        if len(new_neighbours) < neighbours_per_word:
            print(f'Warning: requested {neighbours_per_word} neighbours for word \'{word}\' but only found '
                  f'{len(new_neighbours)} that matched restrictions')

        return new_neighbours

    def filter_neighbours(self, word: str, neighbours: List[str],
                          lemmatize: bool = True, pos: str = wordnet.NOUN,
                          wordnet_filter: bool = True, exclude_secondary_synset_lemmas: bool = False,
                          words_to_exclude: List[str] = None) -> List[str]:
        if words_to_exclude is None:
            words_to_exclude = []

        # Strip words unknown to Wordnet for this part of speech
        if wordnet_filter:
            if pos:
                if exclude_secondary_synset_lemmas:
                    # Get all synsets for each neighbour
                    neighbour_synsets = {neighbour: wordnet.synsets(neighbour, pos=pos) for neighbour in neighbours}
                    # Get the first lemma of each synset for each neighbour
                    first_lemmas = {neighbour: {synset.lemmas()[0].name() for synset in n_synsets}
                                    for neighbour, n_synsets in neighbour_synsets.items()}
                    # Reject neighbours which are not the first lemma of at least one of their synsets
                    # This is harsh but helps remove rare nouns
                    neighbours = [neighbour for neighbour, lemmas in first_lemmas.items() if neighbour in lemmas]
                else:
                    # Exclude neighbours that don't have any synset for this POS in Wordnet
                    neighbours = [neighbour for neighbour in neighbours if wordnet.synsets(neighbour, pos=pos)]

            else:
                neighbours = [neighbour for neighbour in neighbours if wordnet.synsets(neighbour)]

        if lemmatize:
            # Strip plurals and other non-root forms
            neighbours = [self.lemmatizer.lemmatize(neighbour, pos=pos) for neighbour in neighbours]

        # Remove duplicates and copies of original word, and additional words if passed
        new_neighbours = [neighbour for neighbour in OrderedDict.fromkeys(neighbours) if
                          neighbour != word and neighbour not in words_to_exclude]

        return new_neighbours


if __name__ == '__main__':
    test_nouns = ['cat', 'car', 'information']

    # Test nearby words without vocab
    # glove_embeddings = GloveEmbeddings(distance_metric=GloveEmbeddings.COSINE_DISTANCE)
    #
    # nearby_nouns = {noun: glove_embeddings.find_nearby_words(noun, neighbours_per_word=3, lemmatize=True,
    #                                                          wordnet_filter=True, pos=GloveEmbeddings.NOUN,
    #                                                          max_neighbours_to_search=20)
    #                 for noun in test_nouns}
    # for noun, noun_neighbours in nearby_nouns.items():
    #     print(f'{noun}: {", ".join(noun_neighbours)}')

    # Test graph building with vocab
    exp_nouns = read_words('data/combined_experiment_nouns.txt')
    glove_embeddings = GloveEmbeddings(glove_path=GLOVE_6B_100D_PATH,
                                       vocabulary=exp_nouns, distance_metric=GloveEmbeddings.COSINE_SIMILARITY,
                                       max_neighbour_count=5)
    nearby_words = glove_embeddings.get_neighbors('cat', n=3)
    print(nearby_words)
