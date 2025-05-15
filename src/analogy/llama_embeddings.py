import argparse
import csv
import json
from collections import Counter
from typing import List, Union, Iterable

import numpy as np
import pandas as pd
import torch
from torch.nn import Embedding, Module
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, BatchEncoding

from analogy.word_embeddings import WordEmbeddings
from llms.hf_lm_wrapper import ACCELERATE_DEVICE_MAP_OPTIONS
from utils.bigrams import string_to_bigram, extract_ans_from_bigrams
from utils.io import read_bigrams


class LlamaEmbeddings(WordEmbeddings):
    LLAMA_70B_INITIAL_EMBEDDING_PATH = 'output/analogy/Meta-Llama-3-70B-Instruct_initial_embeddings.tsv'
    LLAMA_70B_FINAL_EMBEDDING_PATH = 'output/analogy/Meta-Llama-3-70B-Instruct_final_embeddings.tsv'

    def __init__(self, embedding_path: str = LLAMA_70B_FINAL_EMBEDDING_PATH,
                 distance_metric=WordEmbeddings.COSINE_SIMILARITY,
                 vocabulary: Iterable[str] = None,
                 max_neighbour_count: int = None, min_max_neighbour_similarity: float = None):
        super().__init__(embedding_path,
                         vocabulary=vocabulary,
                         distance_metric=distance_metric,
                         max_neighbour_count=max_neighbour_count,
                         min_max_neighbour_similarity=min_max_neighbour_similarity)

    def load_embeddings(self):
        df = pd.read_table(self.embeddings_path, sep='\t', index_col=0, header=None, quoting=csv.QUOTE_NONE,
                           keep_default_na=False)
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        return df


def load_words(bigram_path: str, extra_analogy_path: str) -> List[str]:
    bigrams = read_bigrams(bigram_path)

    extra_df = pd.read_csv(extra_analogy_path)
    analogy_bigrams = [string_to_bigram(bigram) for bigram in extra_df['AnalogyBigram'].tolist()]

    all_bigrams = bigrams + analogy_bigrams

    adjectives, nouns = extract_ans_from_bigrams(all_bigrams)

    return adjectives + nouns


def load_tokenizer(model_name: str, hf_token: str) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_embedding_layer(model_name: str, device: Union[str, int], hf_token: str) -> Embedding:
    model = load_model(model_name, device, hf_token)
    embeddings = model.model.embed_tokens

    return embeddings


def load_model_headless(model_name: str, device: Union[str, int], hf_token: str) -> Module:
    causal_model = load_model(model_name, device, hf_token)
    inner_model = causal_model.model

    return inner_model


def load_model(model_name: str, device: Union[str, int], hf_token: str) -> Module:
    print('Loading model...')
    if device in ACCELERATE_DEVICE_MAP_OPTIONS:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.float16,
                                                     token=hf_token)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, token=hf_token)
        model.to(device)
    model.eval()

    return model


def batch_words(words: List[str], batch_size: int) -> List[List[str]]:
    return [words[i:i + batch_size] for i in range(0, len(words), batch_size)]


def save_embeddings(words: List[str], tokenizer: PreTrainedTokenizer, embedding: Module, out_path: str,
                    digits: int = 4, full_model: bool = False, batch_size: int = 1):
    token_counts = []
    with open(out_path, "w") as f:
        batched_words = batch_words(sorted(words), batch_size)
        for word_batch in tqdm(batched_words):
            batch_tokens: BatchEncoding = tokenizer(word_batch, return_tensors="pt", padding=True,
                                                    add_special_tokens=False)
            emb_tokens = embedding(batch_tokens["input_ids"])
            if full_model:
                # Embedding layer returns the embeddings directly, but for full model we get a more complex object
                emb_tokens = emb_tokens[0]
            np_tokens = np.nan_to_num(emb_tokens.detach().cpu().numpy())

            # Save tokens - use same format as GloVe for easy processing
            for i in range(len(word_batch)):
                word = word_batch[i]
                word_tokens = np_tokens[i]
                word_ids: List[int] = batch_tokens.encodings[i].word_ids
                start_token_id = word_ids.index(0)  # word_ids are None for padding, 0 for first word
                actual_tokens = word_tokens[start_token_id:]
                avg_tokens = np.average(actual_tokens, axis=0)

                token_count = actual_tokens.shape[0]
                token_counts.append(token_count)

                token_string = '\t'.join(str(round(token.item(), digits)) for token in avg_tokens)
                f.write(f'{word}\t{token_string}\n')

    token_counts = Counter(token_counts)
    print('Token/word counts:', json.dumps(token_counts, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_access_token', type=str, default=None,
                        help='Huggingface access token, if accessing a non-public model '
                             '(e.g. needs a license agreement)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Huggingface model name')
    parser.add_argument('--round_digits', type=int, default=8,
                        help='Digits to round embeddings to')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Optional torch device')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Optional batch size')
    parser.add_argument('--embedding_type', type=str, choices=['initial', 'final'], default='initial')
    parser.add_argument('--main_bigram_path', type=str, required=True,
                        help='Bigrams as TXT file for which to get embeddings of their words')
    parser.add_argument('--additional_bigram_path', type=str,
                        help='Optional path of additional bigrams as CSV file')
    parser.add_argument('--out_dir', type=str, default='output/analogy')
    args = parser.parse_args()
    main_bigram_path = args.main_bigram_path
    analogy_exp_path = args.additional_bigram_path

    short_model_name = args.model_name.split('/')[-1]
    embedding_out_path = f'{args.out_dir}/{short_model_name}_{args.embedding_type}_embeddings.tsv'

    words_to_embed = load_words(main_bigram_path, analogy_exp_path)
    tok = load_tokenizer(args.model_name, args.hf_access_token)
    if args.embedding_type == 'initial':
        emb = load_embedding_layer(args.model_name, args.device, args.hf_access_token)
    else:
        emb = load_model_headless(args.model_name, args.device, args.hf_access_token)
    save_embeddings(words_to_embed, tok, emb, embedding_out_path,
                    digits=args.round_digits, full_model=args.embedding_type == 'final', batch_size=args.batch_size)
