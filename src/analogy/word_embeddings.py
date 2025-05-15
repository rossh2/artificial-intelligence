from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from analogy.word_similarity import WordSimilarities


class WordEmbeddings(WordSimilarities):
    EUCLIDEAN_DISTANCE = 'euclidean'
    COSINE_SIMILARITY = 'cosine'

    def __init__(self, embeddings_path: str, vocabulary: Iterable[str] = None,
                 distance_metric=COSINE_SIMILARITY, max_neighbour_count: int = None,
                 min_max_neighbour_similarity: float = None):
        self.distance_metric = distance_metric

        self.embeddings_path = embeddings_path

        # Set up in self.setup_embeddings
        self.embeddings = None
        self.embeddings_matrix = None

        worst_value = np.inf if distance_metric == self.EUCLIDEAN_DISTANCE else 0.0
        best_value = 0 if distance_metric == self.EUCLIDEAN_DISTANCE else 1.0

        super().__init__(is_distance=(distance_metric == self.EUCLIDEAN_DISTANCE),
                         vocabulary=vocabulary,
                         max_neighbour_count=max_neighbour_count,
                         min_max_neighbour_similarity=min_max_neighbour_similarity,
                         worst_value=worst_value,
                         best_value=best_value)

    def load_embeddings(self) -> pd.DataFrame:
        raise NotImplementedError('Implement in subclass')

    def setup_embeddings(self):
        # Load embeddings
        self.embeddings = self.load_embeddings()

        # Prune words not in vocabulary
        if self.vocabulary:
            self.embeddings = self.embeddings.filter(items=list(self.vocabulary), axis='index')

        self.embeddings_matrix = self.embeddings.to_numpy()

        # Set vocabulary order to match embeddings index
        self.vocabulary = list(self.embeddings.index)

    def build_similarities(self) -> np.ndarray:
        self.setup_embeddings()

        # Calculate distance between each pair
        # Entry at (n, m) represents the similarity between vectors in rows n and m
        if self.distance_metric == self.COSINE_SIMILARITY:
            similarities = cosine_similarity(self.embeddings_matrix)
        else:
            raise ValueError(f'Unsupported distance metric: {self.distance_metric}')

        return similarities

    def get_vector(self, word: str) -> np.ndarray:
        return self.embeddings.loc[word].to_numpy()

    def get_distances(self, vector) -> np.ndarray:
        if self.distance_metric == self.EUCLIDEAN_DISTANCE:
            diff = self.embeddings_matrix - vector
            delta = np.sqrt(np.sum(diff * diff, axis=1))
            return delta
        elif self.distance_metric == self.COSINE_SIMILARITY:
            top = np.dot(self.embeddings_matrix, vector)
            bottom = np.linalg.norm(self.embeddings_matrix, axis=1) * np.linalg.norm(vector)
            delta = top / bottom
            return delta
        else:
            raise NotImplementedError(f'Not implemented distance metric: {self.distance_metric}')

    def find_closest_word(self, vector) -> str:
        delta = self.get_distances(vector)
        i = np.argmin(delta)
        return self.embeddings.iloc[i].name

    def find_closest_words(self, vector: np.ndarray, count: int) -> List[str]:
        delta = self.get_distances(vector)
        if self.sort_descending:
            # Get distances closest to 1, by sorting then reversing
            indices = np.argsort(delta)[::-1]
        else:
            # Get smallest distances (sort from smallest to largest delta)
            indices = np.argsort(delta)

        # Take count many smallest distances
        indices = indices[:count]
        closest_words = [self.embeddings.iloc[i].name for i in indices]
        return closest_words
