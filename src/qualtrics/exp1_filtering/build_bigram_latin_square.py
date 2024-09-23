from random import shuffle
from typing import List, Tuple

from utils.bigrams import build_simple_cross
from utils.io import read_words, write_json, write_bigrams


def group_bigrams_into_blocks(adjectives: List[str], nouns: List[str],
                              blocks_per_group: int = None,
                              print_blocks=False) -> List[List[List[Tuple[str, str]]]]:
    """Group bigrams into groups of blocks of the desired length
    such that no block contains the same adjective or noun twice, and no group contains each noun twice.

    You need the number of nouns to be divisible by the number of adjectives; this forms rotations of the nouns.
    The number of nouns in a rotation is len(nouns)/len(adjectives).
    You need this number to be divisible by the blocks per group so that each rotation's blocks can be divided
        into groups.
    The number of questions per block will be the number of adjectives.

    Returns a list of groups, each group is a list of blocks, each block is a list of bigrams
    For easiest calculation, set blocks_per_group = 1
    """

    questions_per_block = len(adjectives)
    if not blocks_per_group:
        blocks_per_group = len(nouns) / questions_per_block

    # Step 0: shuffle adjectives and nouns so that the groups don't have artifacts (e.g. alphabetical order)
    shuffle(nouns)
    shuffle(adjectives)

    # Step 1: create groups where each noun occurs exactly once
    noun_rotations = [nouns[i:] + nouns[:i] for i in range(len(adjectives))]
    adjective_noun_ratio = len(nouns) / len(adjectives)
    if adjective_noun_ratio.is_integer():
        adjective_noun_ratio = int(adjective_noun_ratio)
    else:
        raise ValueError('Number of nouns is not a multiple of the number of adjectives; '
                         'cannot divide into blocks without repeating adjectives in a block')
    adjectives_repeated = adjectives * adjective_noun_ratio

    rotated_bigrams: List[List[Tuple[str, str]]] = []
    for noun_rotation in noun_rotations:
        rotation_bigrams = []
        for adjective, noun in zip(adjectives_repeated, noun_rotation):
            rotation_bigrams.append((adjective, noun))
        rotated_bigrams.append(rotation_bigrams)

    # Step 2: Divide into blocks
    block_bigrams_by_rotation = []
    for rotation_bigrams in rotated_bigrams:
        block_bigrams = []
        for i in range(adjective_noun_ratio):
            block_bigrams.append(rotation_bigrams[i * questions_per_block:(i + 1) * questions_per_block])
        block_bigrams_by_rotation.append(block_bigrams)

    # Step 3: Flatten blocks according to blocks_per_group
    blocks_per_rotation = adjective_noun_ratio / blocks_per_group
    if blocks_per_rotation.is_integer():
        blocks_per_rotation = int(blocks_per_rotation)
    else:
        # Groups are structured such that each group contains each noun exactly once
        raise ValueError('Cannot divide noun-adjective combinations into blocks without risking repeating a noun')

    grouped_bigrams = []
    for rotation_bigrams in block_bigrams_by_rotation:
        # rotation_bigrams is a list of blocks
        for i in range(blocks_per_rotation):
            grouped_bigrams.append(rotation_bigrams[i * blocks_per_group:(i + 1) * blocks_per_group])

    if print_blocks:
        for i, group_bigrams in enumerate(grouped_bigrams):
            print(f'Group {i + 1}:')
            for j, block_bigrams in enumerate(group_bigrams):
                print(f'  Block {j + 1}:')
                for adjective, noun in block_bigrams:
                    print(f'    {adjective} {noun}')

    return grouped_bigrams


if __name__ == '__main__':
    adjective_path = '../../../bigrams/adjectives.txt'
    noun_set = 'analogy60'  # 48, or 48-96 etc.
    # noun_path = f'../output/filtering_data/frequency_selected_{noun_set}nouns.txt'
    noun_path = '../../../bigrams/experiment_nouns.txt'

    adjs = read_words(adjective_path)
    ns = read_words(noun_path)

    out_path = f'../../../output/filtering_data/grouped_{noun_set}x{len(adjs)}_bigrams.json'

    the_grouped_bigrams = group_bigrams_into_blocks(adjs, ns, blocks_per_group=1, print_blocks=True)

    write_json(the_grouped_bigrams, out_path=out_path)
