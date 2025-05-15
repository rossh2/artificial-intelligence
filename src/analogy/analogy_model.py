import argparse
import itertools
import json
import logging
import math
import statistics
import sys
from collections import defaultdict, Counter
from typing import Tuple, Dict, List, Union, Iterable, Optional

import numpy
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
from typing_extensions import Self

from analogy.glove import GloveEmbeddings
from analogy.llama_embeddings import LlamaEmbeddings
from analogy.word_embeddings import WordEmbeddings
from analogy.word_similarity import WordSimilarities, GroupedSimilarities, find_similar_bigrams
from analogy.wordnet_similarity import WordnetSimilarities
from utils.bigrams import bigram_to_string, string_to_bigram, extract_ans_from_bigrams

HIGH_75_FREQ_SPLIT_MAP = {
    'Zero': 'Zero',
    '25th-50th percentile': 'Low',
    '50th-75th percentile': 'Low',
    '75th-90th percentile': 'High',
    '90th-99th percentile': 'High'
}

HIGH_50_FREQ_SPLIT_MAP = {
    'Zero': 'Zero',
    '25th-50th percentile': 'Low',
    '50th-75th percentile': 'High',
    '75th-90th percentile': 'High',
    '90th-99th percentile': 'High'
}


class RatingDistribution:
    def __init__(self, scale_length: int, ratings: List[int] = None, rating_distribution: Dict[int, float] = None,
                 mean: float = None, sd: float = None,
                 item_label: str = None, split_labels: List[str] = None):
        self.item_label = item_label
        self.split_labels = split_labels
        self.scale_length = scale_length

        if ratings:
            self.ratings = ratings
            self.rating_distribution = self.build_distribution(ratings)
            self.mean = statistics.mean(ratings)
            self.sd = statistics.stdev(ratings) if len(ratings) > 1 else 0
        else:
            self.ratings = []
            self.rating_distribution = rating_distribution
            self.mean = mean
            self.sd = sd

        self.se = 1.96 * self.sd

    def build_distribution(self, ratings_list: List[int]) -> Dict[int, float]:
        rating_counts = Counter(ratings_list)
        total_ratings = len(ratings_list)
        distribution = {i: rating_counts.get(i, 0) / total_ratings for i in range(1, self.scale_length + 1)}
        return distribution

    def __str__(self):
        return f'RatingDistribution[item_label={self.item_label}, mean={round(self.mean, 2)}, sd={round(self.sd, 2)}]'

    def __repr__(self):
        return self.__str__()

    def copy_with_new_labels(self, split_labels: List[str] = None, item_label: str = None) -> Self:
        if not item_label:
            item_label = self.item_label
        if not split_labels:
            split_labels = self.split_labels

        if self.ratings:
            return RatingDistribution(scale_length=self.scale_length, ratings=self.ratings,
                                      item_label=item_label, split_labels=split_labels)
        else:
            return RatingDistribution(scale_length=self.scale_length, rating_distribution=self.rating_distribution,
                                      mean=self.mean, sd=self.sd,
                                      item_label=item_label, split_labels=split_labels)

    def find_split_label_by_value(self, label_options: Iterable[str]) -> str:
        for split_label in self.split_labels:
            if split_label in label_options:
                return split_label

    def find_int_split_label(self) -> int:
        for split_label in self.split_labels:
            try:
                int_value = int(split_label)
                return int_value
            except ValueError:
                continue
        raise ValueError('No integer split label found')

    @classmethod
    def from_mean_variance_csv(cls, csv_path: str, scale_length: int, item_label_column: str,
                               mean_column: str = 'Mean', sd_column: str = 'SD',
                               split_label_columns: List[str] = None) -> List[Self]:
        rating_df = pd.read_csv(csv_path)

        columns = zip(*(rating_df[x] for x in [item_label_column, mean_column, sd_column] + split_label_columns))

        ratings = []
        for row_values in columns:
            item_label, mean, sd = row_values[0:3]
            split_labels = list(row_values[3:])

            # Skip ratings where one of the split labels is empty or nan
            if all([not not split_label and not (isinstance(split_label, float) and math.isnan(split_label))
                    for split_label in split_labels]):
                ratings.append(RatingDistribution(mean=mean, sd=sd,
                                                  item_label=item_label, split_labels=split_labels,
                                                  scale_length=scale_length))
            else:
                logger.warning(f'Skipping rating for item {item_label} since one or more split labels were missing')
        return ratings

    @classmethod
    def from_long_rating_csv(cls, csv_path: str, scale_length: int,
                             item_label_column: str, numeric_rating_column: str,
                             split_label_columns: List[str] = None) -> List[Self]:
        long_rating_df = pd.read_csv(csv_path)
        ratings_by_bigram = long_rating_df.groupby(item_label_column)

        ratings = []
        for item_label, ratings_df in ratings_by_bigram:
            split_labels = [pd.unique(ratings_df[split_label])[0] for split_label in split_label_columns]
            bigram_ratings = ratings_df[numeric_rating_column].to_list()

            # Skip ratings where one of the split labels is empty or nan
            if all([not (isinstance(split_label, str) and split_label == "")
                    and not (isinstance(split_label, float) and math.isnan(split_label))
                    for split_label in split_labels]):
                ratings.append(RatingDistribution(scale_length=scale_length,
                                                  item_label=str(item_label),
                                                  ratings=bigram_ratings,
                                                  split_labels=split_labels))

        return ratings

    @classmethod
    def combine(cls, ratings: List[Self], item_label: str = None,
                weight_by_sd: bool = False, weights: List[float] = None) -> Self:
        if weight_by_sd:
            # Higher the smaller the SD
            weights = cls.calculate_weights_by_sd(ratings)

        # Check all are using the same scale
        assert len(set(rating.scale_length for rating in ratings)) == 1
        scale_length = ratings[0].scale_length

        combined_mean = float(np.average([rating.mean for rating in ratings], weights=weights))
        variances = [rating.sd * rating.sd for rating in ratings]
        combined_sd = np.sqrt(np.average(variances, weights=weights))

        combined_distribution = cls.combine_distributions(ratings, scale_length, weights)
        return RatingDistribution(rating_distribution=combined_distribution,
                                  mean=combined_mean,
                                  sd=combined_sd,
                                  item_label=item_label,
                                  scale_length=scale_length
                                  )

    @classmethod
    def combine_distributions(cls, ratings: List[Self], scale_length: int,
                              weights: List[float] = None) -> Optional[Dict[int, float]]:
        specified_distributions = [rating.rating_distribution for rating in ratings if rating.rating_distribution]
        if not specified_distributions:
            return None

        new_distribution = dict()
        for i in range(1, scale_length + 1):
            new_distribution[i] = float(np.average([dist[i] for dist in specified_distributions], weights=weights))

        return new_distribution

    @classmethod
    def calculate_weights_by_sd(cls, ratings: List[Self]) -> List[float]:
        # Higher the smaller the SD
        return [rating.scale_length - rating.sd for rating in ratings]

    def accurate_within_1sd(self, proposed_rating: Optional[Self], round_to_int=True) -> bool:
        if not proposed_rating or np.isnan(proposed_rating.mean):
            return False
        if round_to_int:
            # Rounding the mean as well dramatically drops performance on subsective adjectives,
            # because the SD is sometimes smaller than 1 and the mean gets rounded down
            return round(self.mean - self.sd) <= proposed_rating.mean <= round(self.mean + self.sd)
        else:
            return self.mean - self.sd < proposed_rating.mean < self.mean + self.sd

    def accurate(self, proposed_rating: Optional[Self], round_to_int=True, max_distance=1) -> bool:
        if not proposed_rating or np.isnan(proposed_rating.mean):
            return False
        # Assumes that self.mean is either ceiling or floor and self.sd = 0
        if round_to_int:
            return abs(self.mean - round(proposed_rating.mean)) <= max_distance
        else:
            return abs(self.mean - proposed_rating.mean) <= max_distance

    def jensen_shannon_divergence(self, proposed_rating: Optional[Self]) -> float:
        if not proposed_rating or not proposed_rating.rating_distribution:
            return 1.0

        d1 = [v for k, v in sorted(self.rating_distribution.items())]
        d2 = [v for k, v in sorted(proposed_rating.rating_distribution.items())]
        with numpy.errstate(invalid='ignore'):
            js = jensenshannon(d1, d2)
        if np.isnan(js):
            # JS was zero but sqrt(zero) = nan
            return 0.0
        else:
            # This is already a float but the typing thinks it's "np.floating"
            return float(js)


class AnalogyModel:
    WITHIN_1_SD_METHOD = 'within-1-SD accuracy'
    ACCURACY_METHOD = 'accuracy'
    JS_METHOD = 'Jensen-Shannon divergence'
    SD_WEIGHTING = 'SD'
    SIMILARITY_WEIGHTING = 'sim'

    def __init__(self, adjective_similarities: WordSimilarities, noun_similarities: WordSimilarities,
                 nearby_adj_count: int = 2, nearby_noun_count: int = 5, max_nearby_bigrams: int = None,
                 max_nearby_distance: float = None,
                 max_distance_nouns_only: bool = True,
                 weight_by: str = None,
                 allow_memorization: bool = False):
        """
        :param nearby_adj/noun_count: the (maximum) number of similar words to consider
        :param max_nearby_distance: if passed, the maximum distance between similar words to consider
        :param max_distance_nouns_only: if passed, maximum distance only applies to nouns
        """
        self.adjective_similarities = adjective_similarities
        self.noun_similarities = noun_similarities

        self.known_ratings: Dict[Tuple[str, str], RatingDistribution] = dict()
        self.fallback_rating: Optional[RatingDistribution] = None
        self.allow_memorization = allow_memorization

        self.nearby_adj_count = nearby_adj_count
        self.nearby_noun_count = nearby_noun_count
        self.max_nearby_known_bigrams = max_nearby_bigrams
        self.nearby_distance = max_nearby_distance
        self.max_distance_nouns_only = max_distance_nouns_only

        self.weight_by = weight_by

    @staticmethod
    def ratings_to_dict(bigram_ratings: List[RatingDistribution]) -> Dict[Tuple[str, str], RatingDistribution]:
        ratings_dict = {string_to_bigram(rating.item_label): rating for rating in bigram_ratings}
        return ratings_dict

    def train(self, bigram_ratings: List[RatingDistribution], fallback_rating: RatingDistribution):
        train_ratings = self.ratings_to_dict(bigram_ratings)
        self.fallback_rating = fallback_rating

        bigrams = train_ratings.keys()
        adjectives, nouns = zip(*bigrams)

        unknown_adjectives = set(adjectives).difference(self.adjective_similarities.vocabulary)
        unknown_nouns = set(nouns).difference(self.noun_similarities.vocabulary)
        known_ratings = {bigram: rating for bigram, rating in train_ratings.items()
                         if bigram[0] not in unknown_adjectives and bigram[1] not in unknown_nouns}

        self.known_ratings = known_ratings

    def tune_hyperparameters(self, bigram_ratings: List[RatingDistribution],
                             max_bigram_used_count: int,
                             max_adj_nearby_count: int = None, max_noun_nearby_count: int = None,
                             max_nearby_distance: float = None, distance_interval: float = None,
                             round_predicted_rating_to_int: bool = True, method=WITHIN_1_SD_METHOD):
        """
        Perform simple grid search over hyperparameters from 0/1/distance_interval to specified values
        """
        prev_logging_level = logger.level
        prev_memorization = self.allow_memorization
        logger.setLevel(logging.INFO)  # Suppress debug output while looping over hyperparameters
        self.allow_memorization = False  # Disable memorization for tuning

        accuracies: Dict[Tuple[int, int, int, float], float] = dict()

        bigram_nearby_counts = np.arange(start=1, stop=max_bigram_used_count + 1, step=1)
        if max_adj_nearby_count:
            adj_nearby_counts = np.arange(start=0, stop=max_adj_nearby_count + 1, step=1)
        else:
            adj_nearby_counts = [self.nearby_adj_count]
        if max_noun_nearby_count:
            noun_nearby_counts = np.arange(start=1, stop=max_noun_nearby_count + 1, step=1)
        else:
            noun_nearby_counts = [self.nearby_noun_count]
        if max_nearby_distance and distance_interval:
            nearby_distances = np.arange(start=distance_interval, stop=max_nearby_distance + distance_interval,
                                         step=distance_interval)
        else:
            nearby_distances = [None]

        hyperparameters_to_check = list(itertools.product(bigram_nearby_counts, adj_nearby_counts, noun_nearby_counts,
                                                          nearby_distances))
        if len(hyperparameters_to_check) > 1:
            logger.info(f'Tuning hyperparameters '
                        f'({"a<=" if len(adj_nearby_counts) > 1 else "a="}{max(adj_nearby_counts)}, '
                        f'{"n<=" if len(noun_nearby_counts) > 1 else "n="}{max(noun_nearby_counts)}, '
                        f'{"k<=" if len(bigram_nearby_counts) > 1 else "k="}{max(bigram_nearby_counts)})...')
            for bigram_nearby_count, adj_nearby_count, noun_nearby_count, nearby_distance \
                    in tqdm(hyperparameters_to_check):
                self.max_nearby_known_bigrams = bigram_nearby_count
                self.nearby_adj_count = adj_nearby_count
                self.nearby_noun_count = noun_nearby_count
                self.nearby_distance = nearby_distance
                acc = self.test(bigram_ratings, round_to_int=round_predicted_rating_to_int, method=method)
                accuracies[(bigram_nearby_count, adj_nearby_count, noun_nearby_count, nearby_distance)] = acc['Total']

            if method in [self.WITHIN_1_SD_METHOD, self.ACCURACY_METHOD]:
                best_parameters = max(accuracies, key=accuracies.get)
            elif method in [self.JS_METHOD]:
                best_parameters = min(accuracies, key=accuracies.get)
            else:
                raise ValueError(f'Unrecognized evaluation method: {method}')

            logger.info(f'Best {method} score of {accuracies[best_parameters]} found using '
                        f'nearby_bigram_count: {best_parameters[0]}, '
                        f'nearby_adj_count: {best_parameters[1]}, nearby_noun_count: {best_parameters[2]} and '
                        f'nearby_distance: {best_parameters[3]}; setting values\n')
            self.max_nearby_known_bigrams = best_parameters[0]
            self.nearby_adj_count = best_parameters[1]
            self.nearby_noun_count = best_parameters[2]
            self.nearby_distance = best_parameters[3]
        else:
            logger.info('Skipping hyperparameter tuning since no range of values provided')

        logger.setLevel(prev_logging_level)
        self.allow_memorization = prev_memorization

    def test(self, bigram_ratings: List[RatingDistribution],
             round_to_int=True, method=WITHIN_1_SD_METHOD) -> Dict[str, float]:
        bigram_rating_dict = self.ratings_to_dict(bigram_ratings)

        predicted_ratings = self.predict(list(bigram_rating_dict.keys()))

        if method in [self.WITHIN_1_SD_METHOD, self.ACCURACY_METHOD]:
            correct_counts = defaultdict(int)
            totals = defaultdict(int)
            for predicted, actual in zip(predicted_ratings, bigram_rating_dict.values()):
                if method == self.WITHIN_1_SD_METHOD:
                    accurate = actual.accurate_within_1sd(predicted, round_to_int)
                else:
                    accurate = actual.accurate(predicted, round_to_int)
                if accurate:
                    correct_counts['Total'] += 1
                    correct_counts['-'.join(actual.split_labels)] += 1
                    for split_label in actual.split_labels:
                        correct_counts[split_label] += 1
                totals['Total'] += 1
                totals['-'.join(actual.split_labels)] += 1
                for split_label in actual.split_labels:
                    totals[split_label] += 1

            accuracies = dict()
            for key in totals:
                accuracies[key] = correct_counts[key] / totals[key]
            return accuracies

        elif method == self.JS_METHOD:
            bigrams = []
            dists = []
            divergences = []
            summed_js_divergences = defaultdict(float)
            counts = defaultdict(int)
            for predicted, actual in zip(predicted_ratings, bigram_rating_dict.values()):
                divergence = actual.jensen_shannon_divergence(predicted)
                bigrams.append(actual.item_label)
                dists.append(predicted.rating_distribution if predicted else
                             {i: 0 for i in range(actual.scale_length)})
                divergences.append(divergence)
                counts['Total'] += 1
                counts['-'.join(actual.split_labels)] += 1
                summed_js_divergences['Total'] += divergence
                summed_js_divergences['-'.join(actual.split_labels)] += divergence
                for split_label in actual.split_labels:
                    counts[split_label] += 1
                    summed_js_divergences[split_label] += divergence

            mean_js_divergences = dict()
            for name in counts.keys():
                mean_js_divergences[name] = summed_js_divergences[name] / counts[name]

            js_df = pd.DataFrame.from_dict({'Bigram': bigrams, 'JSDivergence': divergences})
            dist_df = pd.DataFrame.from_records(dists).add_prefix('Rating')
            js_df = pd.concat([js_df, dist_df], axis='columns')
            logger.debug('JS Divergence CSV:')
            logger.debug(js_df.to_csv(path_or_buf=None, index=False, lineterminator='\n'))

            return mean_js_divergences
        else:
            raise ValueError(f'Unknown method {method}')

    def predict(self, bigrams: List[Union[Tuple[str, str], str]]) -> List[RatingDistribution]:
        results = []
        if not isinstance(bigrams[0], tuple):
            bigrams = [string_to_bigram(bigram) for bigram in bigrams]

        for bigram in bigrams:
            rating = self.get_rating_by_analogy(bigram)
            results.append(rating)
        return results

    def get_rating_by_analogy(self, bigram: Tuple[str, str]) -> Optional[RatingDistribution]:
        adjective, noun = bigram

        logger.debug(f'Estimating rating for bigram "{adjective} {noun}"')

        if self.allow_memorization and bigram in self.known_ratings:
            return self.known_ratings[bigram]

        nearby_bigrams: List[Tuple[Tuple[str, str], float]]
        nearby_bigrams = find_similar_bigrams(self.adjective_similarities, self.noun_similarities, bigram,
                                              max_adj_neighbours=self.nearby_adj_count,
                                              max_noun_neighbours=self.nearby_noun_count,
                                              min_word_similarity=self.nearby_distance,
                                              by_sim=(self.weight_by == self.SIMILARITY_WEIGHTING),
                                              return_similarities=True)

        rated_bigrams = [bigram for bigram, distance in nearby_bigrams if bigram in self.known_ratings]
        logger.debug(f'Found nearby bigrams: '
                     f'{", ".join([bigram_to_string(bigram) for bigram, distance in nearby_bigrams])}')
        logger.debug(f'of which the following have known ratings: '
                     f'{", ".join([bigram_to_string(bigram) for bigram in rated_bigrams])}')

        nearby_ratings = [(self.known_ratings[bigram], distance) for bigram, distance in nearby_bigrams
                          if bigram in self.known_ratings]

        if self.max_nearby_known_bigrams:
            if len(nearby_ratings) < self.max_nearby_known_bigrams:
                logger.debug(
                    f'Warning: fewer than {self.max_nearby_known_bigrams} nearby bigrams found for {adjective} {noun}')
            # Already sorted by distance, so take the n closest
            nearby_ratings = nearby_ratings[:self.max_nearby_known_bigrams]

        if not nearby_ratings:
            if self.fallback_rating:
                logger.debug(
                    f'Warning: no known nearby bigrams for {adjective} {noun}, returning provided fallback rating')
                return self.fallback_rating.copy_with_new_labels(item_label=bigram_to_string(bigram))
            else:
                logger.debug(f'Warning: no known nearby bigrams for {adjective} {noun}, returning no rating')
                return None

        if self.weight_by == self.SIMILARITY_WEIGHTING:
            weights = [distance for rating, distance in nearby_ratings]
            # In order to rank by similarity, adjective and noun similarity must already be ordered the same
            if not self.noun_similarities.sort_descending:
                # 0 is best and 1 / infinity is worst
                weights = [max(1.0, max(weights)) - weight for weight in weights]
        else:
            weights = None
        proposed_rating = RatingDistribution.combine([rating for rating, distance in nearby_ratings],
                                                     bigram_to_string(bigram), weights=weights,
                                                     weight_by_sd=self.weight_by == self.SD_WEIGHTING)
        return proposed_rating


def split_ratings_by_top_n(ratings: List[RatingDistribution], adjectives: List[str], n: int,
                           train_key: str = 'Top-N',
                           non_zero_test_key: str = None) -> Dict[str, List[RatingDistribution]]:
    if not non_zero_test_key:
        # Can get passed as None
        non_zero_test_key = 'Non-' + train_key

    ratings_by_class = defaultdict(list)
    ratings_by_adj = partition_ratings_by_split(ratings, target_splits=adjectives)
    for adj, adj_ratings in ratings_by_adj.items():
        sorted_ratings = sorted([(rating.find_int_split_label(), rating) for rating in adj_ratings],
                                key=lambda x: x[0], reverse=True)
        for i, (count, rating) in enumerate(sorted_ratings):
            # Remove adjective split_label and update frequency split label
            other_split_labels = [label for label in rating.split_labels if label != count and label != adj]
            if i < n:
                new_split_labels = [train_key] + other_split_labels
                rating_copy = rating.copy_with_new_labels(new_split_labels)
                ratings_by_class[train_key].append(rating_copy)
            else:
                split_key = 'Zero' if count == 0 else non_zero_test_key
                new_split_labels = [split_key] + other_split_labels
                rating_copy = rating.copy_with_new_labels(new_split_labels)
                ratings_by_class[split_key].append(rating_copy)

    return ratings_by_class


def partition_ratings_by_split(ratings: List[RatingDistribution], target_splits: List[str] = None,
                               split_map: Dict[str, str] = None) -> Dict[str, List[RatingDistribution]]:
    ratings_by_class = defaultdict(list)

    if not split_map and not target_splits:
        raise ValueError('Must specify one of target_splits or split_map, since ratings may have many split labels')
    if split_map:
        target_splits = split_map.keys()

    for rating in ratings:
        target_label = rating.find_split_label_by_value(target_splits)
        if split_map:
            new_label = split_map[target_label]
            other_split_labels = [label for label in rating.split_labels if label != target_label]
            new_split_labels = [new_label] + other_split_labels
            rating_copy = rating.copy_with_new_labels(new_split_labels)
            ratings_by_class[new_label].append(rating_copy)
        else:
            ratings_by_class[target_label].append(rating)

    return ratings_by_class


def get_train_test_splits(ratings: List[RatingDistribution], adjectives: List[str],
                          split_map: Dict[str, str] = None, split_top_n: int = None,
                          train_split_name: str = None, test_split_name: str = None):
    if split_map:
        split_ratings = partition_ratings_by_split(ratings, split_map=split_map)
        if not train_split_name:
            train_split_name = 'High'
    elif split_top_n:
        if not train_split_name:
            train_split_name = 'Top-N'
        split_ratings = split_ratings_by_top_n(ratings, adjectives, split_top_n,
                                               train_key=train_split_name,
                                               non_zero_test_key=test_split_name)

    else:
        raise ValueError('Must pass one of split_map or split_top_n')

    train = split_ratings[train_split_name]
    val = train
    test = []
    for key in split_ratings:
        if key != train_split_name:
            test += split_ratings[key]

    return train, val, test


def setup_glove_model(adjectives: List[str], nouns: List[str],
                      max_adj_candidate_neighbour_count: int, max_noun_candidate_neighbour_count: int,
                      min_max_similarity: float = None,
                      max_used_neighbour_count: int = None,
                      initial_nearby_adj_count: int = None, initial_nearby_noun_count: int = None,
                      weight_by: str = None, allow_memorization: bool = False) -> AnalogyModel:
    logger.info('\nLoading GloVe embeddings...')

    if not initial_nearby_adj_count:
        initial_nearby_adj_count = max_adj_candidate_neighbour_count
    if not initial_nearby_noun_count:
        initial_nearby_noun_count = max_noun_candidate_neighbour_count
    if not max_used_neighbour_count:
        max_used_neighbour_count = max(max_adj_candidate_neighbour_count, max_noun_candidate_neighbour_count)

    a_embeddings = GloveEmbeddings(distance_metric=WordEmbeddings.COSINE_SIMILARITY,
                                   vocabulary=adjectives,
                                   max_neighbour_count=max_adj_candidate_neighbour_count,
                                   min_max_neighbour_similarity=min_max_similarity)
    n_embeddings = GloveEmbeddings(distance_metric=WordEmbeddings.COSINE_SIMILARITY,
                                   vocabulary=nouns,
                                   max_neighbour_count=max_noun_candidate_neighbour_count,
                                   min_max_neighbour_similarity=min_max_similarity)

    logger.info('Setting up GloVe model...')
    model = AnalogyModel(adjective_similarities=a_embeddings, noun_similarities=n_embeddings,
                         weight_by=weight_by,
                         nearby_adj_count=initial_nearby_adj_count, nearby_noun_count=initial_nearby_noun_count,
                         max_nearby_bigrams=max_used_neighbour_count,
                         allow_memorization=allow_memorization)

    return model


def setup_llama_model(adjectives: List[str], nouns: List[str],
                      max_adj_candidate_neighbour_count: int, max_noun_candidate_neighbour_count: int,
                      use_final_embeddings: bool = True,
                      min_max_similarity: float = None,
                      max_used_neighbour_count: int = None,
                      initial_nearby_adj_count: int = None, initial_nearby_noun_count: int = None,
                      weight_by: str = None, allow_memorization: bool = False) -> AnalogyModel:
    logger.info('\nLoading Llama embeddings...')

    if not initial_nearby_adj_count:
        initial_nearby_adj_count = max_adj_candidate_neighbour_count
    if not initial_nearby_noun_count:
        initial_nearby_noun_count = max_noun_candidate_neighbour_count
    if not max_used_neighbour_count:
        max_used_neighbour_count = max(max_adj_candidate_neighbour_count, max_noun_candidate_neighbour_count)

    embedding_path = LlamaEmbeddings.LLAMA_70B_FINAL_EMBEDDING_PATH if use_final_embeddings \
        else LlamaEmbeddings.LLAMA_70B_INITIAL_EMBEDDING_PATH

    a_embeddings = LlamaEmbeddings(distance_metric=WordEmbeddings.COSINE_SIMILARITY,
                                   embedding_path=embedding_path,
                                   vocabulary=adjectives,
                                   max_neighbour_count=max_adj_candidate_neighbour_count,
                                   min_max_neighbour_similarity=min_max_similarity)
    n_embeddings = LlamaEmbeddings(distance_metric=WordEmbeddings.COSINE_SIMILARITY,
                                   embedding_path=embedding_path,
                                   vocabulary=nouns,
                                   max_neighbour_count=max_noun_candidate_neighbour_count,
                                   min_max_neighbour_similarity=min_max_similarity)

    logger.info('Setting up Llama embedding model...')
    model = AnalogyModel(adjective_similarities=a_embeddings, noun_similarities=n_embeddings,
                         weight_by=weight_by,
                         nearby_adj_count=initial_nearby_adj_count, nearby_noun_count=initial_nearby_noun_count,
                         max_nearby_bigrams=max_used_neighbour_count,
                         allow_memorization=allow_memorization)

    return model


def setup_wordnet_model(adj_class_df: pd.DataFrame, nouns: List[str], max_synsets: int,
                        max_candidate_neighbour_count: int, min_max_similarity: float = None,
                        max_used_neighbour_count: int = None,
                        initial_nearby_adjective_count: int = None, initial_nearby_noun_count: int = None,
                        weight_by: str = None, allow_memorization: bool = False) -> AnalogyModel:
    logger.info('\n\nLoading WordNet similarities...')

    if not initial_nearby_noun_count:
        initial_nearby_noun_count = max_candidate_neighbour_count
    if not max_used_neighbour_count:
        max_used_neighbour_count = max_candidate_neighbour_count

    # Problem: WordNet doesn't implement a useful adjective distance
    # As a proxy we can use adjective class
    grouped_adjectives = adj_class_df.groupby('AdjectiveClass')['Adjective'].apply(list).values.tolist()
    a_embeddings = GroupedSimilarities(grouped_vocabulary=grouped_adjectives)
    n_embeddings = WordnetSimilarities(pos=WordnetSimilarities.NOUN,
                                       vocabulary=nouns,
                                       max_synsets=max_synsets,
                                       max_neighbour_count=max_candidate_neighbour_count,
                                       min_max_neighbour_similarity=min_max_similarity)

    logger.info('Setting up WordNet model...')
    model = AnalogyModel(adjective_similarities=a_embeddings,
                         noun_similarities=n_embeddings,
                         weight_by=weight_by,
                         nearby_adj_count=initial_nearby_adjective_count,
                         nearby_noun_count=initial_nearby_noun_count,
                         max_nearby_bigrams=max_used_neighbour_count,
                         allow_memorization=allow_memorization)

    return model


def extract_adjectives_nouns(ratings: List[RatingDistribution]) -> Tuple[List[str], List[str]]:
    bigrams = set(string_to_bigram(rating.item_label) for rating in ratings)
    adjectives, nouns = extract_ans_from_bigrams(bigrams)
    return adjectives, nouns


def evaluate_model_ootb(model: AnalogyModel, train_ratings: List[RatingDistribution],
                        test_ratings: List[RatingDistribution], method=AnalogyModel.WITHIN_1_SD_METHOD):
    train_accuracies = model.test(train_ratings, method=method)
    logger.info(f'\nAccuracy on training bigrams with {method} method:')
    for split, accuracy in train_accuracies.items():
        logger.info(f'{split}: {accuracy}')

    test_accuracies = model.test(test_ratings, method=method)
    logger.info(f'\nAccuracy on test bigrams with {method} method:')
    for split, accuracy in test_accuracies.items():
        logger.info(f'{split}: {accuracy}')

    total_accuracies = model.test(train_ratings + test_ratings, method=method)
    logger.info(f'\nAccuracy on all bigrams with {method} method:')
    for split, accuracy in total_accuracies.items():
        logger.info(f'{split}: {accuracy}')
    logger.info('\n')


def get_context_bigrams() -> List[str]:
    df = pd.read_csv('data/adjective_contexts.csv')
    return df['Bigram'].unique().tolist()


def evaluate_context(model: AnalogyModel, context_ratings: List[RatingDistribution],
                     freq_splits: List[str]) -> Dict[str, Dict[str, float]]:
    subsective_context_ratings = [RatingDistribution(mean=5.0, sd=0, scale_length=5,
                                                     item_label=rating.item_label,
                                                     split_labels=[label for label in rating.split_labels if
                                                                   label in freq_splits])
                                  for rating in context_ratings]
    privative_context_ratings = [RatingDistribution(mean=1.0, sd=0, scale_length=5,
                                                    item_label=rating.item_label,
                                                    split_labels=[label for label in rating.split_labels if
                                                                  label in freq_splits])
                                 for rating in context_ratings]

    subs_acc = model.test(subsective_context_ratings, method=AnalogyModel.ACCURACY_METHOD)
    priv_acc = model.test(privative_context_ratings, method=AnalogyModel.ACCURACY_METHOD)

    averages = {}
    for key in subs_acc:
        averages[key] = np.mean([subs_acc[key], priv_acc[key]])

    return {
        'Subsective context': subs_acc,
        'Privative context': priv_acc,
        'Total': averages
    }


def evaluate_as_context_baseline(model: AnalogyModel, ootb_ratings: List[RatingDistribution],
                                 split_map: Dict[str, str]):
    context_bigrams = get_context_bigrams()
    context_ratings = [rating for rating in ootb_ratings if rating.item_label in context_bigrams]

    context_accuracies = evaluate_context(model, context_ratings, list(split_map.values()))

    logger.info('Accuracy on context set / context metric:')
    for context, accuracies in context_accuracies.items():
        for split, accuracy in accuracies.items():
            logger.info(f'{context}: {split}: {accuracy}')


def train_and_tune_model(model: AnalogyModel, train_dists: List[RatingDistribution],
                         val_dists: List[RatingDistribution] = None,
                         max_bigram_used_count: int = None,
                         max_adj_neighbour_count: int = None, max_noun_neighbour_count: int = None,
                         fallback_rating: Optional[RatingDistribution] = None,
                         eval_method=AnalogyModel.WITHIN_1_SD_METHOD):
    model.train(train_dists, fallback_rating)

    if val_dists:
        model.tune_hyperparameters(val_dists,
                                   max_bigram_used_count=max_bigram_used_count,
                                   max_adj_nearby_count=max_adj_neighbour_count,
                                   max_noun_nearby_count=max_noun_neighbour_count,
                                   max_nearby_distance=None,
                                   method=eval_method)


def configure_logging(cfg: dict, log_out_dir: str):
    combined_logger = logging.getLogger()
    combined_logger.setLevel(logging.DEBUG)
    log_file = f'{log_out_dir}/analogy_model'
    log_file += f'_{cfg["model_type"].lower()}'
    log_file += f'_{cfg["max_candidate_adj_neighbours"]}a'
    log_file += f'_{cfg["max_candidate_noun_neighbours"]}n'
    if cfg['max_used_neighbours']:
        log_file += f'_{cfg["max_used_neighbours"]}b'
    if cfg['use_human_analogy_bigrams']:
        log_file += '_with-ap'
    if cfg['fallback_rating']:
        log_file += '_fallback'
    if cfg['weight_ratings_by']:
        log_file += f'_weighted-{cfg["weight_ratings_by"]}'
    if cfg['balance_training_by_adjective_n']:
        log_file += f'_bal{cfg["balance_training_by_adjective_n"]}'
    if cfg['50th_pct_as_high']:
        log_file += '_50high'
    if cfg['memorize_training_set']:
        log_file += '_memo'
    log_file += '.log'
    file_log_handler = logging.FileHandler(log_file, mode='w')
    file_log_handler.setLevel(logging.DEBUG)
    combined_logger.addHandler(file_log_handler)
    stream_log_handler = logging.StreamHandler(sys.stdout)
    stream_log_handler.setLevel(logging.INFO)
    combined_logger.addHandler(stream_log_handler)

    return combined_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run analogy model with configurable parameters.")

    # Config parameters
    parser.add_argument('--max_candidate_noun_neighbours', type=int, default=100,
                        help="Sets the max nouns for the hyperparameter search. "
                             "If using analogy bigrams, many candidate bigrams aren't known, "
                             "so need to set this high to find enough (set max_used low to compensate)")
    parser.add_argument('--max_candidate_adj_neighbours', type=int, default=1,
                        help="Sets the max adjectives for the hyperparameter search. "
                             "Realistically this usually gets chosen as around 1-2. "
                             "Set to 0 to use the same adjective as the input")
    parser.add_argument('--max_used_neighbours', type=int, default=5, choices=[1, 5])
    parser.add_argument('--max_wordnet_synsets', type=int, default=2)
    parser.add_argument('--tune_max_nouns', action='store_true',
                        help="Not so relevant if max_used_neighbours is set low (1-5)")
    parser.add_argument('--tune_max_adjs', action='store_true')
    parser.add_argument('--fallback_rating', action='store_true')
    parser.add_argument('--weight_ratings_by', default=AnalogyModel.SIMILARITY_WEIGHTING,
                        choices=[None, AnalogyModel.SIMILARITY_WEIGHTING, AnalogyModel.SD_WEIGHTING])
    parser.add_argument('--memorize_training_set', action='store_true',
                        help="Doesn't apply to hyperparameter tuning, only at test time")
    parser.add_argument('--use_human_analogy_bigrams', action='store_true')
    parser.add_argument('--50th_pct_as_high', action='store_true')
    parser.add_argument('--balance_training_by_adjective_n', type=int, default=None,
                        help="Set to None or the integer n bigrams per adjective to choose")
    parser.add_argument('--model_type', default='GloVe',
                        choices=['GloVe', 'WordNet', 'Llama-initial', 'Llama-final'])

    # Paths
    parser.add_argument('', required=True,
                        help='Path to the bigram rating CSV file')
    parser.add_argument('--analogy_bigram_rating_path', required=True,
                        help='Path to the analogy bigram CSV file')
    parser.add_argument('--adjective_class_path', required=True,
                        help='Path to the adjective class CSV file')
    parser.add_argument('--log_out_dir',
                        default='output/analogy/analogy_model/',
                        help='Path to directory in which to save log files')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    config = {
        'max_candidate_noun_neighbours': args.max_candidate_noun_neighbours,
        'max_candidate_adj_neighbours': args.max_candidate_adj_neighbours,
        'max_used_neighbours': args.max_used_neighbours,
        'max_wordnet_synsets': args.max_wordnet_synsets,
        'tune_max_nouns': args.tune_max_nouns,
        'tune_max_adjs': args.tune_max_adjs,
        'fallback_rating': args.fallback_rating,
        'weight_ratings_by': args.weight_ratings_by,
        'memorize_training_set': args.memorize_training_set,
        'use_human_analogy_bigrams': args.use_human_analogy_bigrams,
        '50th_pct_as_high': args.__dict__['50th_pct_as_high'],  # Can't start variable name with number
        'balance_training_by_adjective_n': args.balance_training_by_adjective_n,
        'model_type': args.model_type,
    }

    logger = configure_logging(config, args.log_out_dir)

    adj_classes = pd.read_csv(args.adjective_class_path)

    logger.info(f'Configuration: {json.dumps(config, indent=2, default=str)}')

    logger.info('Loading human ratings...')
    frequency_split_labels = ['Count', 'Adjective', 'AdjectiveClass'] \
        if config['balance_training_by_adjective_n'] else ['CoarseFrequency', 'AdjectiveClass']
    exp_ratings = RatingDistribution.from_long_rating_csv(args.bigram_rating_path,
                                                          scale_length=5,
                                                          item_label_column='Bigram',
                                                          numeric_rating_column='NumRating',
                                                          split_label_columns=frequency_split_labels)

    all_adjectives, all_nouns = extract_adjectives_nouns(exp_ratings)
    logger.info(f'...loaded {len(exp_ratings)} bigrams ({len(all_adjectives)} adjectives, {len(all_nouns)} nouns)')

    if config['balance_training_by_adjective_n']:
        train_set, val_set, test_set = get_train_test_splits(exp_ratings, all_adjectives,
                                                             split_top_n=config['balance_training_by_adjective_n'],
                                                             train_split_name='Top-N')
        freq_split_map = None
    else:
        freq_split_map = HIGH_50_FREQ_SPLIT_MAP if config['50th_pct_as_high'] else HIGH_75_FREQ_SPLIT_MAP
        train_set, val_set, test_set = get_train_test_splits(exp_ratings, all_adjectives,
                                                             split_map=freq_split_map, train_split_name='High')

        logger.debug(f'Mapping frequencies to training splits: {json.dumps(freq_split_map, indent=2)}')
    train_ratings_per_adjective = Counter(string_to_bigram(rating.item_label)[0] for rating in train_set)
    train_nouns = set(string_to_bigram(rating.item_label)[1] for rating in train_set)
    logger.debug(f'Building training set with {len(train_set)} ratings, {len(train_nouns)} nouns '
                 f'and the following number of bigrams per adjective: '
                 f'{json.dumps(train_ratings_per_adjective, indent=2)}')

    if config['use_human_analogy_bigrams']:
        extra_train_set = RatingDistribution.from_long_rating_csv(args.analogy_bigram_rating_path,
                                                                  scale_length=5,
                                                                  item_label_column='AnalogyBigram',
                                                                  numeric_rating_column='NumRating',
                                                                  split_label_columns=['AdjectiveClass'])
        known_bigrams = set(rating.item_label for rating in exp_ratings)
        extra_train_set = [rating for rating in extra_train_set if rating.item_label not in known_bigrams]

        ana_adjectives, ana_nouns = extract_adjectives_nouns(extra_train_set)
        all_adjectives = list(set(all_adjectives).union(ana_adjectives))
        all_nouns = list(set(all_nouns).union(ana_nouns))

        ana_adj_classes = pd.DataFrame.from_dict({'Adjective': ana_adjectives,
                                                  'AdjectiveClass': range(len(ana_adjectives))})
        adj_classes = pd.concat([adj_classes, ana_adj_classes], ignore_index=True)
        logger.info(f'...loaded additional {len(extra_train_set)} bigrams ' +
                    f'({len(ana_adjectives)} adjectives, {len(ana_nouns)} nouns) from analogy prompting experiment')
    else:
        extra_train_set = []

    if config['model_type'] == 'GloVe':
        an_model = setup_glove_model(all_adjectives, all_nouns,
                                     max_adj_candidate_neighbour_count=config['max_candidate_adj_neighbours'],
                                     max_noun_candidate_neighbour_count=config['max_candidate_noun_neighbours'],
                                     max_used_neighbour_count=config['max_used_neighbours'],
                                     weight_by=config['weight_ratings_by'],
                                     allow_memorization=config['memorize_training_set'])
        logger.info('Loading GloVe model...')
    elif config['model_type'].startswith('Llama'):
        if not config['model_type'].endswith('final') and not config['model_type'].endswith('initial'):
            raise ValueError('Invalid Llama model type')
        final_embeddings = config['model_type'].endswith('final')
        an_model = setup_llama_model(all_adjectives, all_nouns,
                                     use_final_embeddings=final_embeddings,
                                     max_adj_candidate_neighbour_count=config['max_candidate_adj_neighbours'],
                                     max_noun_candidate_neighbour_count=config['max_candidate_noun_neighbours'],
                                     max_used_neighbour_count=config['max_used_neighbours'],
                                     weight_by=config['weight_ratings_by'],
                                     allow_memorization=config['memorize_training_set'])
        logger.info('Loading Llama model...')
    elif config['model_type'] == 'WordNet':
        an_model = setup_wordnet_model(adj_classes, all_nouns,
                                       max_synsets=config['max_wordnet_synsets'],
                                       max_candidate_neighbour_count=config['max_candidate_noun_neighbours'],
                                       max_used_neighbour_count=config['max_used_neighbours'],
                                       weight_by=config['weight_ratings_by'],
                                       allow_memorization=config['memorize_training_set'])
        logger.info('Loading Wordnet model...')
    else:
        raise ValueError(f'Unrecognized model type {config["model_type"]}')

    logger.info('Training model with "Within 1 SD" metric...')
    max_noun_tuning_count = None if not config['tune_max_nouns'] else config['max_candidate_noun_neighbours']
    max_adj_tuning_count = None if not config['tune_max_adjs'] else config['max_candidate_adj_neighbours']
    train_and_tune_model(an_model, train_set + extra_train_set,
                         val_dists=val_set,
                         fallback_rating=config['fallback_rating'],
                         max_bigram_used_count=config['max_used_neighbours'],
                         max_adj_neighbour_count=max_adj_tuning_count,
                         max_noun_neighbour_count=max_noun_tuning_count,
                         eval_method=AnalogyModel.WITHIN_1_SD_METHOD)

    logger.info('Evaluating with "Within 1 SD" metric')
    evaluate_model_ootb(an_model, train_set, test_set,
                        method=AnalogyModel.WITHIN_1_SD_METHOD)

    if config['balance_training_by_adjective_n']:
        logger.info('Skipping evaluating model trained on "Within 1 SD" metric as context baseline '
                    'since splits not defined')
    else:
        logger.info('Evaluating model trained on "Within 1 SD" metric as context baseline')
        evaluate_as_context_baseline(an_model, train_set + test_set, freq_split_map)

    logger.info('Training model with Jensen-Shannon divergence (over distributions)...')
    train_and_tune_model(an_model, train_set, val_set,
                         fallback_rating=config['fallback_rating'],
                         max_bigram_used_count=config['max_used_neighbours'],
                         max_adj_neighbour_count=max_adj_tuning_count,
                         max_noun_neighbour_count=max_noun_tuning_count,
                         eval_method=AnalogyModel.JS_METHOD)

    logger.info('Evaluating with Jensen-Shannon divergence (over distributions)...')
    evaluate_model_ootb(an_model, train_set, test_set, method=AnalogyModel.JS_METHOD)
