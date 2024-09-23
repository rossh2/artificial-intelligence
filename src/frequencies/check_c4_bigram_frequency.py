import argparse
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from itertools import islice
from math import ceil
from multiprocessing import cpu_count
from typing import Dict, List, Tuple, Iterable

from datasets import load_dataset, load_dataset_builder
from tqdm import tqdm

from utils.io import write_json, read_words, read_bigrams
from utils.bigrams import bigram_to_string, build_simple_cross, extract_ans_from_bigrams

WORD_COUNT = 'WORD_COUNT'
DOC_COUNT = 'DOC_COUNT'
TOTAL_BYTES = 'TOTAL_BYTES'

NOUN_KEY = 'nouns'
ADJECTIVE_KEY = 'adjectives'
BIGRAM_KEY = 'bigrams'
METADATA_KEY = 'metadata'

DEFAULT_PROCESSES = 2
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DOC_GROUP_SIZE = 100


def chunk_iterable(iterable: Iterable, chunk_size: int):
    it = iter(iterable)

    def take():
        while True:
            yield islice(it, chunk_size)

    return take()


def get_document_counts(adjectives: List[str], nouns: List[str], bigram_strings: List[str], document_text: str,
                        doc_count: int = 1) -> Dict[str, Counter[str, int]]:
    bigram_output: Dict[str, int] = {bigram: 0 for bigram in bigram_strings}
    adj_output: Dict[str, int] = {adjective: 0 for adjective in adjectives}
    noun_output: Dict[str, int] = {noun: 0 for noun in nouns}

    lower_document_text = document_text.lower()

    for adjective in adjectives:
        adj_count = lower_document_text.count(adjective)
        adj_output[adjective] += adj_count

    for noun in nouns:
        noun_count = lower_document_text.count(noun)
        noun_output[noun] += noun_count

    for bigram_string in bigram_strings:
        bigram_count = lower_document_text.count(bigram_string)
        bigram_output[bigram_string] += bigram_count

    approx_document_word_count = len(document_text.split())
    approx_document_bytes = len(document_text)

    metadata = Counter({
        WORD_COUNT: approx_document_word_count,
        TOTAL_BYTES: approx_document_bytes,
        DOC_COUNT: doc_count
    })

    return {
        ADJECTIVE_KEY: Counter(adj_output),
        NOUN_KEY: Counter(noun_output),
        BIGRAM_KEY: Counter(bigram_output),
        METADATA_KEY: metadata,
    }


def group_and_count_documents(adjectives: List[str], nouns: List[str], bigrams: List[str],
                              docs: List[Dict]) -> Dict[str, Counter[str, int]]:
    concat_text = ''
    for doc in docs:
        concat_text += doc['text']
        concat_text += '\n\n'

    return get_document_counts(adjectives, nouns, bigrams, document_text=concat_text, doc_count=len(docs))


def get_bigram_counts(adjectives: List[str], nouns: List[str], bigrams: List[Tuple[str, str]],
                      doc_limit: float, process_count: int, chunk_size: int, doc_group_size: int,
                      save_interval: int, download_dataset: bool, shuffle: bool) -> Dict[str, Counter[str, int]]:
    doc_count = get_c4_info()
    doc_limit = doc_limit if doc_limit < float('inf') else doc_count
    chunk_limit = ceil(doc_limit / (chunk_size * doc_group_size))  # Each chunk contains groups of documents
    items_per_process = chunk_size / process_count  # Each item is a group of documents
    if round(items_per_process) != items_per_process:
        print(f'Warning: chunk size {chunk_size} '
              f'does not divide evenly across {process_count} processes')
        items_per_process = round(items_per_process)

    c4_dataset = load_dataset('c4', 'en', split='train', streaming=not download_dataset)
    if shuffle and download_dataset and doc_limit < doc_count:
        tqdm.write('Shuffling dataset...')
        # Shuffling downloads the entire dataset!
        # No need to shuffle if we're using all the examples anyway
        c4_dataset = c4_dataset.shuffle(seed=42)

    bigram_strings = [bigram_to_string(bigram) for bigram in bigrams]
    # Insert a space before adjectives so e.g. performer =/= former; will miss small set of punctuation e.g. "former"
    # Do this just once here to save on computation
    adjective_strings = [' ' + adjective for adjective in adjectives]
    noun_strings = [' ' + noun for noun in nouns]

    adjective_counter_list = []
    noun_counter_list = []
    bigram_counter_list = []
    metadata_counter_list = []
    counter_lists = [adjective_counter_list, noun_counter_list, bigram_counter_list, metadata_counter_list]

    overall_adjective_counts = Counter({adj: 0 for adj in adjective_strings})
    overall_noun_counts = Counter({noun: 0 for noun in noun_strings})
    overall_bigram_counts = Counter({bigram: 0 for bigram in bigram_strings})
    overall_metadata = Counter({
        WORD_COUNT: 0,
        TOTAL_BYTES: 0,
        DOC_COUNT: 0
    })
    overall_counters = [overall_adjective_counts, overall_noun_counts, overall_bigram_counts, overall_metadata]

    grouped_c4 = chunk_iterable(c4_dataset, chunk_size=doc_group_size)
    chunked_c4 = chunk_iterable(grouped_c4, chunk_size=chunk_size)
    bound_get_counts = partial(group_and_count_documents, adjective_strings, noun_strings, bigram_strings)

    tqdm.write(f'Spinning up {process_count} processes ({cpu_count()} available cores); '
               f'chunk size per processor: {int(items_per_process)}\n')
    time.sleep(0.01)  # Otherwise print statement gets tangled up with subsequent tqdm call

    with ProcessPoolExecutor(max_workers=process_count) as executor:
        for i in tqdm(range(chunk_limit), desc='chunks', position=0, leave=True, file=sys.stdout):
            next_c4 = next(chunked_c4)
            future_counts = [executor.submit(bound_get_counts, list(doc_group))
                             for doc_group in next_c4]

            for future_counter in tqdm(as_completed(future_counts),
                                       total=chunk_size, desc='docs/chunk', position=1, leave=False, file=sys.stdout):
                try:
                    doc_counter_dict = future_counter.result()
                    adjective_counter_list.append(doc_counter_dict[ADJECTIVE_KEY])
                    noun_counter_list.append(doc_counter_dict[NOUN_KEY])
                    bigram_counter_list.append(doc_counter_dict[BIGRAM_KEY])
                    metadata_counter_list.append(doc_counter_dict[METADATA_KEY])
                except TimeoutError as e:
                    print(e)

            if i % save_interval == 0 and i != 0:
                for counter_list, overall_counter in zip(counter_lists, overall_counters):
                    new_total_counts = sum(counter_list, Counter())
                    overall_counter.update(new_total_counts)

                save_counts({
                    ADJECTIVE_KEY: overall_adjective_counts,
                    NOUN_KEY: overall_noun_counts,
                    BIGRAM_KEY: overall_bigram_counts,
                    METADATA_KEY: overall_metadata
                }, doc_limit, clean_and_sort=False)
                # Free up memory
                adjective_counter_list, noun_counter_list, bigram_counter_list, metadata_counter_list = [], [], [], []

            if overall_metadata[DOC_COUNT] > doc_limit:
                # Sometimes we exceed the intended document limit by quite a bit due to the chunk size * group_size,
                # so stop when done
                print(f'Exceeded doc_limit {doc_limit} (current total: {overall_metadata[DOC_COUNT]} documents) '
                      f'after {i+1} out of {chunk_limit} chunks, stopping early')
                break

    for counter_list, overall_counter in zip(counter_lists, overall_counters):
        new_total_counts = sum(counter_list, Counter())
        overall_counter.update(new_total_counts)

    return {
        ADJECTIVE_KEY: overall_adjective_counts,
        NOUN_KEY: overall_noun_counts,
        BIGRAM_KEY: overall_bigram_counts,
        METADATA_KEY: overall_metadata
    }


def get_c4_info() -> int:
    c4_dataset_info = load_dataset_builder('c4', 'en')
    doc_count = c4_dataset_info.info.splits['train'].num_examples
    byte_count = c4_dataset_info.info.splits['train'].num_bytes
    print(f'C4 dataset: {doc_count} total documents, {format_as_gb(byte_count)} total')
    # Expected approx. 300 words/document, so 100,000,000,000 words (1e11)

    return doc_count


def sort_counts_descending(counts: Dict[str, int]) -> Dict[str, int]:
    """
    Also cleans up any spaces at edges of keys
    This is done in a single pass through the dictionary (rather than as two separate steps)
    for efficiency
    """

    return {key.strip(): value for key, value in
            sorted(counts.items(), key=lambda x: (-x[1], x[0]))}


def format_as_gb(byte_count: int) -> str:
    return f'{byte_count / (1024 ** 3):.2f}GB'


def save_counts(all_counts: Dict[str, Counter[str, int]], doc_limit: float, clean_and_sort=True):
    doc_count = all_counts[METADATA_KEY][DOC_COUNT]
    formatted_doc_count = f'{doc_count}'
    if doc_count < doc_limit < float('inf'):
        formatted_doc_count += f'of{int(doc_limit)}'
    bigram_count = len(all_counts[BIGRAM_KEY])
    out_path = f'{args.out_dir}/c4_bigram_counts_{formatted_doc_count}docs_{bigram_count}bigrams.json'

    if clean_and_sort:
        sorted_adjective_counts = sort_counts_descending(all_counts[ADJECTIVE_KEY])
        sorted_noun_counts = sort_counts_descending(all_counts[NOUN_KEY])
        sorted_bigram_counts = sort_counts_descending(all_counts[BIGRAM_KEY])

        write_json({
            'doc_count': all_counts[METADATA_KEY][DOC_COUNT],
            'total_word_count': all_counts[METADATA_KEY][WORD_COUNT],
            'byte_count': all_counts[METADATA_KEY][TOTAL_BYTES],
            'adjective_counts': sorted_adjective_counts,
            'noun_counts': sorted_noun_counts,
            'bigram_counts': sorted_bigram_counts,

        }, out_path=out_path)
    else:
        write_json({
            'doc_count': all_counts[METADATA_KEY][DOC_COUNT],
            'total_word_count': all_counts[METADATA_KEY][WORD_COUNT],
            'byte_count': all_counts[METADATA_KEY][TOTAL_BYTES],
            'adjective_counts': all_counts[ADJECTIVE_KEY],
            'noun_counts': all_counts[NOUN_KEY],
            'bigram_counts': all_counts[BIGRAM_KEY],

        }, out_path=out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processors', default=DEFAULT_PROCESSES, type=int,
                        help='The number of processes to spawn. '
                             'Note that 3 of these appear to be used for other purposes (not on the main task)')
    parser.add_argument('--chunk_size', default=DEFAULT_CHUNK_SIZE, type=int,
                        help='number of documents passed to each process at once - the larger the better')
    parser.add_argument('--doc_group_size', default=DEFAULT_DOC_GROUP_SIZE, type=int,
                        help='number of documents concatenated into one pseudo-document - the larger the better')
    parser.add_argument('--doc_limit', default=float('inf'), type=float,
                        help='maximum number of documents to search (c4 contains 364,868,892 total documents)')
    parser.add_argument('--adjectives', type=str, required=False,
                        help='text file containing the adjectives to count, '
                             'one adjective per line. '
                             'Bigrams will be assembled by crossing all adjectives with all nouns, unless passed.')
    parser.add_argument('--nouns', type=str, required=False,
                        help='text file containing the nouns to count, '
                             'one noun per line.'
                             'Bigrams will be assembled by crossing all adjectives with all nouns, unless passed.')
    parser.add_argument('--bigrams', type=str, required=False,
                        help='text file containing the bigrams to count, one bigram per line. You must pass either '
                             'this argument or adjectives and nouns, which will be crossed to form the bigrams.')
    parser.add_argument('--skip_bigram_counts', action='store_true', default=False,
                        help='Skip counting bigrams and only count adjective and noun frequencies')
    parser.add_argument('--skip_an_counts', action='store_true', default=False,
                        help='Skip counting adjectives and nouns and only count bigram frequencies')
    parser.add_argument('--out_dir', type=str, help='directory for output file', required=True)
    parser.add_argument('--save_interval', default=3, type=int,
                        help='Save counts to file after every n chunks '
                             '(useful in case of errors or process termination)')
    parser.add_argument('--download', action='store_true', default=False,
                        help='Pass this flag to download the whole dataset instead of streaming. '
                             'Note that streaming disables shuffling, as shuffling the dataset requires downloading '
                             'the whole dataset, even if streaming, so this is not practical if you only want to '
                             'stream a small portion of the dataset.')
    parser.add_argument('--shuffle', action='store_true', default=False,
                        help='Pass this flag to shuffle the dataset before sampling. '
                             'Note that streaming disables shuffling, as shuffling the dataset requires downloading '
                             'the whole dataset, even if streaming, so this is not practical if you only want to '
                             'stream a small portion of the dataset.'
                        )

    args = parser.parse_args()
    print(args)

    input_adjectives, input_nouns, input_bigrams = [], [], []
    if args.adjectives and args.nouns:
        input_adjectives = read_words(args.adjectives)
        input_nouns = read_words(args.nouns)
    if args.bigrams:
        input_bigrams = read_bigrams(args.bigrams)

    if not input_bigrams:
        if not (args.adjectives and args.nouns):
            raise ValueError('Must pass either bigrams or adjectives plus nouns')
        if args.skip_bigram_counts:
            print(f'Skipping building bigrams, just counting adjectives and nouns')
        else:
            input_bigrams = build_simple_cross(adjectives=input_adjectives, nouns=input_nouns)
            print(
                f'Built {len(input_bigrams)} bigrams from {len(input_adjectives)} adjectives and {len(input_nouns)} nouns')

    if not input_adjectives and not input_nouns:
        if args.skip_an_counts:
            print(f'Skipping extracting adjectives and nouns, just counting bigrams')
        else:
            input_adjectives, input_nouns = extract_ans_from_bigrams(input_bigrams)
            print(
                f'Extracted {len(input_adjectives)} adjectives and {len(input_nouns)} nouns from {len(input_bigrams)} bigrams')

    if args.skip_bigram_counts and args.skip_an_counts:
        raise ValueError('Skipping bigrams and adjectives and nouns: not counting anything')

    final_counts = get_bigram_counts(
        adjectives=input_adjectives if not args.skip_an_counts else [],
        nouns=input_nouns if not args.skip_an_counts else [],
        bigrams=input_bigrams,
        doc_limit=args.doc_limit,
        process_count=args.processors, chunk_size=args.chunk_size,
        doc_group_size=args.doc_group_size,
        save_interval=args.save_interval,
        download_dataset=args.download,
        shuffle=args.shuffle,
    )

    print(f'Byte count: {format_as_gb(final_counts[METADATA_KEY][TOTAL_BYTES])}')
    print(f'Document count: {final_counts[METADATA_KEY][DOC_COUNT]}')
    print(f'Total word count (approximate): {final_counts[METADATA_KEY][WORD_COUNT]}')

    save_counts(final_counts, args.doc_limit, clean_and_sort=True)
