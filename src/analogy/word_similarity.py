import random
from collections import defaultdict
from statistics import mean
from typing import Iterable, List, Union, Tuple, Dict

import igraph as ig
import numpy as np


class WordSimilarities:
    def __init__(self, is_distance: bool = False, vocabulary: Iterable[str] = None,
                 max_neighbour_count: int = None, min_max_neighbour_similarity: float = None,
                 worst_value: float = 0, best_value: float = 1):
        self.sort_descending = not is_distance
        self.vocabulary = self.clean_vocabulary(vocabulary)
        self.restricted_vocabulary = vocabulary is not None

        self.max_neighbour_count = max_neighbour_count

        # Upper threshold if is distance (i.e. lower is better),
        # lower threshold if is similarity (i.e. higher is better)
        self.min_max_neighbour_similarity = min_max_neighbour_similarity
        self.worst_value = worst_value
        self.best_value = best_value

        self.distance_graph = None
        if self.restricted_vocabulary:
            # Not enough memory to build this over all possible words!
            self.distance_graph = self.build_similarity_graph()

    @staticmethod
    def clean_vocabulary(vocabulary: Iterable[str]):
        cleaned_vocabulary = []
        for word in vocabulary:
            if ' ' in word:
                # Noun-noun compound, take root
                print(f'Warning: splitting compound {word} by spaces and taking root (last word)')
                word = word.split(' ')[-1]
            cleaned_vocabulary.append(word)

        return cleaned_vocabulary

    def build_similarities(self) -> np.ndarray:
        raise NotImplementedError()

    def build_similarity_graph(self) -> ig.Graph:
        similarities = self.build_similarities()
        assert similarities.shape == (len(self.vocabulary), len(self.vocabulary))

        edges = []

        # Assumes that vocabulary indexes similarity matrix
        word_count = len(self.vocabulary)
        for i in range(word_count):
            word1 = self.vocabulary[i]
            for j in range(i + 1, word_count):
                word2 = self.vocabulary[j]
                if i != j:
                    edges.append((word1, word2, similarities[i, j]))
                    edges.append((word2, word1, similarities[j, i]))

        # Build graph (every vertex connected to every vertex)
        distance_graph = ig.Graph.TupleList(edges, directed=True, weights=True)

        # Prune edges with distance too great or exceeding max number of edges
        for word_node in distance_graph.vs:
            sorted_edges = self.get_sorted_edges(word_node)
            if self.max_neighbour_count:
                edges_to_cut = [e.index for e in sorted_edges[self.max_neighbour_count:]]
                distance_graph.delete_edges(edges_to_cut)
                sorted_edges = sorted_edges[:self.max_neighbour_count]
            if self.min_max_neighbour_similarity:
                i = 0
                while i < len(sorted_edges):
                    if self.sort_descending:
                        # High is better and comes earlier in list
                        if sorted_edges[i]['weight'] < self.min_max_neighbour_similarity:
                            break
                    else:
                        # Low is better and comes earlier in list
                        if sorted_edges[i]['weight'] > self.min_max_neighbour_similarity:
                            break
                if i < len(sorted_edges):
                    edges_to_cut = [e.index for e in sorted_edges[i:]]
                    distance_graph.delete_edges(edges_to_cut)
        return distance_graph

    def get_sorted_edges(self, node: ig.Vertex) -> List[ig.Edge]:
        edges = node.out_edges()
        # Sorted best to worst distance
        sorted_edges = sorted(edges, key=lambda e: e['weight'], reverse=self.sort_descending)
        return sorted_edges

    def get_neighbors(self, word: str, n: int = None, min_or_max_value: float = None,
                      return_similarities: bool = False, exclude_zero_similarity: bool = False,
                      sample_equal_similarities: bool = False) -> Union[List[str], List[Tuple[str, float]]]:
        if not self.distance_graph or word not in self.vocabulary:
            raise ValueError(f'Word "{word}" not in vocabulary or vocabulary not specified')
        if n is None:
            if not self.max_neighbour_count:
                raise ValueError('Must specify n or set class max_neighbour_count')
            n = self.max_neighbour_count

        word_node = self.distance_graph.vs.find(name=word)
        sorted_edges = self.get_sorted_edges(word_node)

        if min_or_max_value:
            sorted_edges = self.get_edges_by_weight(sorted_edges, min_or_max_value, inclusive=True)
        elif exclude_zero_similarity:
            sorted_edges = self.get_edges_by_weight(sorted_edges, self.worst_value, inclusive=False)

        if sample_equal_similarities:
            cutoff_similarity = sorted_edges[n - 1]['weight']
            neighbour_edges = self.get_edges_by_weight(sorted_edges, cutoff_similarity, inclusive=False)
            equal_edges = self.get_edges_by_weight(sorted_edges, cutoff_similarity, inclusive=True)
            equal_edges = equal_edges[len(neighbour_edges):]
            edges_desired = n - len(neighbour_edges)
            neighbour_edges.extend(random.sample(equal_edges, edges_desired))
        else:
            neighbour_edges = sorted_edges[:n]

        if return_similarities:
            neighbours = [(e.target_vertex['name'], e['weight']) for e in neighbour_edges]
        else:
            neighbours = [e.target_vertex['name'] for e in neighbour_edges]
        return neighbours

    def get_edges_by_weight(self, sorted_edges: List[ig.Edge], min_or_max_value: float,
                            inclusive: bool = True) -> List[ig.Edge]:
        i = 0
        while i < len(sorted_edges):
            if self.sort_descending:
                # High is better and comes earlier in list
                if not inclusive and sorted_edges[i]['weight'] <= min_or_max_value:
                    break
                elif sorted_edges[i]['weight'] < min_or_max_value:
                    break
            else:
                # Low is better and comes earlier in list
                if not inclusive and sorted_edges[i]['weight'] >= min_or_max_value:
                    break
                elif sorted_edges[i]['weight'] > min_or_max_value:
                    break
            i += 1
        if i < len(sorted_edges):
            return sorted_edges[:i]
        else:
            return sorted_edges

    def get_mean_neighbour_similarity(self, word: str, neighbour_count: int = None):
        neighbours = self.get_neighbors(word, neighbour_count, return_similarities=True)
        similarities = [sim for word, sim in neighbours]
        mean_distance = mean(similarities)
        return mean_distance


class GroupedSimilarities(WordSimilarities):
    def __init__(self, grouped_vocabulary: List[Iterable[str]], ingroup_similarity: float = 0.5):
        self.grouped_vocabulary, self.vocab_index = self.build_vocab_index(grouped_vocabulary)
        flat_vocabulary = [word for group in grouped_vocabulary for word in group]

        self.ingroup_similarity = ingroup_similarity
        super().__init__(is_distance=False, vocabulary=flat_vocabulary, max_neighbour_count=len(flat_vocabulary))

    @staticmethod
    def build_vocab_index(grouped_vocabulary: List[Iterable[str]]) -> Tuple[Dict[int, List[str]], Dict[str, int]]:
        grouped_vocabulary = {i: list(words) for i, words in enumerate(grouped_vocabulary)}
        vocab_index = dict()
        for group_index, words in grouped_vocabulary.items():
            for word in words:
                vocab_index[word] = group_index
        return grouped_vocabulary, vocab_index

    def build_similarities(self) -> np.ndarray:
        vocab_size = len(self.vocabulary)
        similarities = np.zeros(shape=(vocab_size, vocab_size))

        for i in range(vocab_size):
            word1 = self.vocabulary[i]
            group_index = self.vocab_index[word1]
            group_words = self.grouped_vocabulary[group_index]
            for j in range(i, vocab_size):
                word2 = self.vocabulary[j]
                if word1 == word2:
                    sim = 1
                elif word2 in group_words:
                    sim = self.ingroup_similarity
                else:
                    sim = 0
                similarities[i, j] = sim
                similarities[j, i] = sim

        return similarities


class DictSimilarities(WordSimilarities):
    DEFAULT_SIMILARITY = 0.5

    def __init__(self, similarity_dict: Dict[str, List[Tuple[List[str], float]]]):
        self.similarity_dict = defaultdict(dict)
        for word, neighbours_with_sims in similarity_dict.items():
            for neighbours, sim in neighbours_with_sims:
                for neighbour in neighbours:
                    self.similarity_dict[word][neighbour] = sim

        flat_vocabulary = similarity_dict.keys()

        super().__init__(is_distance=False, vocabulary=flat_vocabulary, max_neighbour_count=len(flat_vocabulary))

    def build_similarities(self) -> np.ndarray:
        vocab_size = len(self.vocabulary)
        similarities = np.zeros(shape=(vocab_size, vocab_size))

        for i in range(vocab_size):
            word1 = self.vocabulary[i]
            word_neighbours = self.similarity_dict[word1]
            for j in range(vocab_size):
                word2 = self.vocabulary[j]
                if word1 == word2:
                    sim = 1
                elif word2 in word_neighbours:
                    sim = word_neighbours[word2]
                else:
                    sim = 0
                similarities[i, j] = sim

        return similarities


def find_similar_bigrams(adjective_similarities: WordSimilarities, noun_similarities: WordSimilarities,
                         bigram: Tuple[str, str], available_bigrams: List[Tuple[str, str]] = None,
                         by_sim: bool = False, adjective_change_penalty: float = 0.0,
                         max_adj_neighbours: int = None, max_noun_neighbours: int = None,
                         max_rank: int = None, min_word_similarity: float = None,
                         sample_equal_similarities: bool = False,
                         return_similarities: bool = False) -> Union[List[Tuple[str, str]], List[Tuple[Tuple[str, str], float]]]:
    if by_sim and adjective_similarities.sort_descending != noun_similarities.sort_descending:
        raise ValueError('Can\'t rank by similarity if adjectives and nouns don\'t both use similarity')
    sort_descending = adjective_similarities.sort_descending

    adjective, noun = bigram
    ranked_similar_adjectives = adjective_similarities.get_neighbors(adjective,
                                                                     n=max_adj_neighbours,
                                                                     min_or_max_value=min_word_similarity,
                                                                     return_similarities=True,
                                                                     sample_equal_similarities=sample_equal_similarities,
                                                                     exclude_zero_similarity=True)
    id_value = 1.0 if sort_descending else 0.0
    ranked_similar_adjectives = [(adjective, id_value)] + ranked_similar_adjectives
    ranked_similar_nouns = noun_similarities.get_neighbors(noun,
                                                           n=max_noun_neighbours,
                                                           min_or_max_value=min_word_similarity,
                                                           return_similarities=True,
                                                           sample_equal_similarities=sample_equal_similarities,
                                                           exclude_zero_similarity=True)
    ranked_similar_nouns = [(noun, id_value)] + ranked_similar_nouns

    # Using min_or_max_value instead
    # if min_word_similarity:
    #     if sort_descending:
    #         ranked_similar_adjectives = [(adj, sim) for adj, sim in ranked_similar_adjectives if
    #                                      sim >= min_word_similarity]
    #         ranked_similar_nouns = [(noun, sim) for noun, sim in ranked_similar_nouns if sim >= min_word_similarity]
    #
    #     else:
    #         ranked_similar_adjectives = [(adj, sim) for adj, sim in ranked_similar_adjectives if
    #                                      sim <= min_word_similarity]
    #         ranked_similar_nouns = [(noun, sim) for noun, sim in ranked_similar_nouns if sim <= min_word_similarity]

    bigrams_by_rank: Dict[float, List[Tuple[str, str]]] = defaultdict(list)
    for noun_rank, (noun, noun_sim) in enumerate(ranked_similar_nouns):
        for adjective_rank, (adjective, adjective_sim) in enumerate(ranked_similar_adjectives):
            if (adjective, noun) != bigram and (not available_bigrams or (adjective, noun) in available_bigrams):
                if by_sim:
                    # Assumes similarities are comparable scale and both sorted ascending or both descending
                    combined_sim = adjective_sim + noun_sim
                    if adjective_change_penalty and adjective != bigram[0]:
                        if sort_descending:
                            # 1+1=2 is best, make it worse by decreasing
                            combined_sim = max(combined_sim - adjective_change_penalty, 0)
                        else:
                            # 0 is best, make it worse by increasing
                            combined_sim += adjective_change_penalty
                    bigrams_by_rank[combined_sim].append((adjective, noun))
                else:
                    combined_rank = adjective_rank + noun_rank
                    if adjective_change_penalty and adjective != bigram[0]:
                        combined_rank += adjective_change_penalty
                    bigrams_by_rank[combined_rank].append((adjective, noun))

    ranked_bigrams = []
    for i in sorted(bigrams_by_rank.keys(), reverse=by_sim and sort_descending):
        if not max_rank or i <= max_rank:
            if return_similarities:
                ranked_bigrams.extend((bigram, i) for bigram in bigrams_by_rank[i])
            else:
                ranked_bigrams.extend(bigrams_by_rank[i])

    return ranked_bigrams


# Loosely scaled to be compatible with WupSim, which is >0.5 for siblings
HANDCRAFTED_ADJECTIVE_SIMILARITIES = DictSimilarities({
    'artificial': [(['fake', 'false'], 0.75), (['counterfeit', 'knockoff'], 0.5)],
    'counterfeit': [(['knockoff'], 0.9), (['fake', 'false'], 0.75), (['artificial'], 0.5)],
    'fake': [(['artificial', 'counterfeit', 'false', 'knockoff'], 0.75)],
    'false': [(['fake'], 0.9), (['counterfeit', 'knockoff', 'artificial'], 0.75)],
    'knockoff': [(['counterfeit'], 0.9), (['fake'], 0.75)],
    'former': [(['artificial', 'counterfeit', 'fake', 'false', 'knockoff'], 0.5)],
    'homemade': [(['artificial', 'fake', 'false'], 0.8), (['tiny', 'multicolored'], 0.75),
                 (['useful', 'illegal', 'unimportant'], 0.5)],
    'useful': [(['homemade', 'tiny', 'illegal', 'unimportant', 'multicolored'], 0.5)],
    'tiny': [(['homemade', 'useful', 'illegal', 'unimportant', 'multicolored'], 0.5)],
    'illegal': [(['homemade', 'tiny', 'useful', 'unimportant', 'multicolored'], 0.5)],
    'unimportant': [(['homemade', 'tiny', 'illegal', 'useful', 'multicolored'], 0.5)],
    'multicolored': [(['homemade', 'tiny', 'illegal', 'unimportant', 'useful'], 0.5)],
})
