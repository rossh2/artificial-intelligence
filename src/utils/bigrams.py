from typing import List, Tuple, Iterable


def bigram_to_string(bigram: Tuple[str, str]) -> str:
    return ' '.join(bigram)


def string_to_bigram(bigram: str) -> Tuple[str, str]:
    words = bigram.split(' ')
    if len(words) == 2:
        adj, noun = words
    else:
        adj = words[0]
        noun = ' '.join(words[1:])
    return adj, noun


def build_simple_cross(adjectives: List[str], nouns: List[str]) -> List[Tuple[str, str]]:
    an_pairs: List[Tuple[str, str]] = []
    for adjective in adjectives:
        for noun in nouns:
            an_pairs.append((adjective, noun))

    return sorted(an_pairs)


def extract_ans_from_bigrams(bigrams: Iterable[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    adjectives, nouns = zip(*bigrams)

    unique_adjectives = list(sorted(set(adjectives)))
    unique_nouns = list(sorted(set(nouns)))

    return unique_adjectives, unique_nouns
