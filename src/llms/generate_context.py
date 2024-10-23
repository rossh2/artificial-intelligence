import argparse
import json

import pandas as pd

from lm_isa_experiment.config_utils import update_scorer_config_from_args
from lm_isa_experiment.lm_generation_wrapper import LMGenerationWrapper
from utils.build_question import build_bigram_questions, add_question_column
from utils.io import get_filename

from context_generation_wrapper import ContextGenerationWrapper


def get_model_generations(bigram_df: pd.DataFrame, lm_wrapper: ContextGenerationWrapper,
                          dump_out_path: str = None) -> pd.DataFrame:
    initial_prompts = bigram_df['InitialPrompt'].to_list()
    follow_up_prompts = bigram_df['FollowUpPrompt'].to_list()
    triple_follow_up_prompts = bigram_df['TripleFollowUpPrompt'].to_list()
    prompts = list(zip(initial_prompts, follow_up_prompts, triple_follow_up_prompts))

    # Returns list of lists, each list has response_count items
    prompt_responses = lm_wrapper.generate_multiple_responses(prompts, dump_out_path=dump_out_path)

    column_names = [f'Context{j + 1}' for j in range(lm_wrapper.response_count)]
    response_df = pd.DataFrame(prompt_responses, columns=column_names)

    combined_df = pd.concat([bigram_df, response_df], axis=1)
    return combined_df


def build_bigrams_with_prompt_columns(bigram_path: str, adjectives_path: str,
                                      mass_count_path: str, config: dict) -> pd.DataFrame:
    bigrams_with_prompts = build_bigram_questions(bigram_path=bigram_path,
                                                  adjectives_path=adjectives_path,
                                                  mass_count_path=mass_count_path,
                                                  question_format_string=config['question_format_string'],
                                                  question_column_name='InitialPrompt')
    mass_count_df = pd.read_csv(mass_count_path)
    bigrams_with_prompts = add_question_column(bigrams_with_prompts,
                                               mass_count_df,
                                               question_format_string=config['follow_up_question_format_string'],
                                               column_name='FollowUpPrompt')
    bigrams_with_prompts = add_question_column(bigrams_with_prompts,
                                               mass_count_df,
                                               question_format_string=config[
                                                   'triple_follow_up_question_format_string'],
                                               column_name='TripleFollowUpPrompt')
    return bigrams_with_prompts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bigrams', type=str, required=True,
                        help='Path to a TXT file with a list of bigrams')
    parser.add_argument('--target_adjectives', type=str, default='data/adjectives.txt',
                        help='Path to a TXT file listing adjectives to restrict experiment bigrams to')
    parser.add_argument('--mass_count', type=str, default='data/mass_count_nouns.csv',
                        help='Path to a CSV file listing mass/count status of nouns')
    parser.add_argument('--scorer_config', type=str, required=True, nargs='+',
                        help='Path(s) to one or more JSON configs for the LM wrapper '
                             '(can pass multiple if doing prompt engineering or other comparison '
                             'on the same set of bigrams)')
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='Maximum number of tokens to generate in response')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Model generation temperature')
    parser.add_argument('--response_count', type=int, default=None,
                        help='Expected number of responses per prompt')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Path to the output directory for the model responses (written as a CSV)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Optional Huggingface model name, if not passed in scorer config')
    parser.add_argument('--device', type=str, default=None,
                        help='Optional torch device, if not passed in scorer config')
    parser.add_argument('--precision', type=int, default=None,
                        help='Optional quantization/precision, if not passed in scorer config')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Optional batch size, if not passed in scorer config.')
    parser.add_argument('--hf_access_token', type=str, default=None,
                        help='Huggingface access token, if accessing a non-public model '
                             '(e.g. needs a license agreement)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed to set for NumPy and PyTorch')

    args = parser.parse_args()
    print_args = vars(args).copy()
    print_args.pop('hf_access_token', None)  # Do not log the access token
    print('Python arguments:')
    print(print_args)

    gen_wrapper = None
    bigrams_with_questions = None
    config_count = len(args.scorer_config)
    for i, scorer_config_path in enumerate(args.scorer_config):

        with open(scorer_config_path) as config_file:
            scorer_config = json.load(config_file)

        # Print scorer config before updating so access token is not printed;
        # if any other updates happen, warning will be printed
        print(f'Scorer config {i + 1}/{config_count}:')
        print(scorer_config)
        update_scorer_config_from_args(scorer_config, args)

        print(f'Building questions from bigrams...')
        bigrams_with_questions = build_bigrams_with_prompt_columns(bigram_path=args.bigrams,
                                                                   adjectives_path=args.target_adjectives,
                                                                   mass_count_path=args.mass_count,
                                                                   config=scorer_config)

        if gen_wrapper is None:
            print(f'Loading model {scorer_config["model_name"]}...')
            gen_wrapper = ContextGenerationWrapper.from_config(scorer_config)
        else:
            print(f'Updating prompting configuration for {scorer_config["model_name"]} ...')
            gen_wrapper.update_from_config(scorer_config)

        short_model_name = scorer_config["model_name"].split('/')[-1]
        bigrams_name = get_filename(args.bigrams).replace('_', '-')
        out_path = f'{args.out_dir}/generations_{bigrams_name}_{short_model_name}_{scorer_config["config_nickname"]}.csv'
        dump_out_path = f'{args.out_dir}/dump_{scorer_config["config_nickname"]}.json'

        print(f'Getting predictions for {len(bigrams_with_questions)} questions...')
        model_responses = get_model_generations(bigrams_with_questions, gen_wrapper, dump_out_path=dump_out_path)

        print(f'Writing full model responses to {out_path}')
        model_responses.to_csv(out_path, index=False)
