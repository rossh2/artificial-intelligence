import json
from typing import List, Tuple, Dict, Union


def read_lines(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as word_file:
        lines = word_file.readlines()
        return [line.strip() for line in lines]


def write_lines(lines: List[str], out_path: str):
    with open(out_path, 'w', encoding='utf-8') as out_file:
        for line in lines:
            out_file.write(line + '\n')


def read_words(path: str) -> List[str]:
    return read_lines(path)


def write_words(words: List[str], out_path: str):
    return write_lines(words, out_path)


def read_bigrams(path: str) -> List[Tuple[str, str]]:
    with open(path, 'r', encoding='utf-8') as bigram_file:
        lines = bigram_file.readlines()
        bigrams = [line.strip().split('\t') for line in lines]
        for bigram in bigrams:
            assert len(bigram) == 2
        bigrams = [(bigram[0], bigram[1]) for bigram in bigrams]

        return bigrams


def write_bigrams(bigrams: List[Tuple[str, str]], out_path: str):
    with open(out_path, 'w', encoding='utf-8') as out_file:
        for word1, word2 in bigrams:
            out_file.write(f'{word1}\t{word2}\n')


def read_json(path: str) -> Union[List, Dict]:
    with open(path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)


def write_json(obj: Union[List, Dict], out_path: str, sort_keys=False):
    with open(out_path, 'w', encoding='utf-8') as out_file:
        json.dump(obj, out_file, sort_keys=sort_keys, indent=4)


def write_text_file(text: str, out_path: str):
    with open(out_path, 'w', encoding='utf-8') as out_file:
        out_file.write(text)


def get_filename(file_path: str) -> str:
    """
    Return just the name of the file without directories or file type extension
    """
    return file_path.split('/')[-1].split('.')[0]
