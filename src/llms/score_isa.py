import argparse
import json
from collections import defaultdict

import pandas as pd

from lm_isa_experiment.config_utils import update_scorer_config_from_args
from lm_isa_experiment.qa_scorer import LMQuestionAnswerScorer
from utils.build_question import build_bigram_questions
from utils.io import get_filename


def predict_model_responses(bigram_df: pd.DataFrame, scorer: LMQuestionAnswerScorer,
                            dump_path: str = None) -> pd.DataFrame:
    prompts = bigram_df['Question'].to_list()
    responses = scorer.get_response_surprisals(prompts, dump_out_path=dump_path)
    predictions = scorer.get_predicted_responses(responses)

    response_dict = defaultdict(list)
    for prompt_responses in responses:
        for response, surprisal in prompt_responses:
            response_dict[response + 'Surprisal'].append(surprisal)

    response_df = bigram_df.copy()
    response_df['PredictedResponse'] = predictions
    response_df = response_df.assign(**response_dict)
    return response_df


def print_metrics(response_df: pd.DataFrame):
    response_percentages = response_df['PredictedResponse'].value_counts(normalize=True)
    print("Response percentages:")
    print(response_percentages)

    response_by_adj_percentages = response_df.groupby(['Adjective'])['PredictedResponse'].value_counts(normalize=True)
    print("Response percentages by adjective:")
    print(response_by_adj_percentages)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bigrams', type=str, required=True,
                        help='Path to a TXT file with a list of bigrams, or a CSV file with bigrams + contexts'
                             'if --context is passed')
    parser.add_argument('--target_adjectives', type=str, default='data/adjectives.txt',
                        help='Path to a TXT file listing adjectives to restrict experiment bigrams to')
    parser.add_argument('--mass_count', type=str, default='data/mass_count_nouns.csv',
                        help='Path to a CSV file listing mass/count status of nouns')
    parser.add_argument('--context', type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help='Pass --context if the bigrams are a CSV containing bigrams and contexts;'
                             'make sure this matches the scorer config')
    parser.add_argument('--scorer_config', type=str, required=True, nargs='+',
                        help='Path(s) to one or more JSON configs for the LM wrapper '
                             '(can pass multiple if doing prompt engineering or other comparison '
                             'on the same set of bigrams)')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Path to the output directory for the model responses (written as a CSV)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Optional Huggingface model name, if not passed in scorer config')
    parser.add_argument('--device', type=str, default=None,
                        help='Optional torch device, if not passed in scorer config')
    parser.add_argument('--precision', type=int, default=None,
                        help='Optional quantization/precision, if not passed in scorer config')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Optional batch size, if not passed in scorer config')
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

    qa_scorer = None
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
        bigrams_with_questions = build_bigram_questions(bigram_path=args.bigrams,
                                                        adjectives_path=args.target_adjectives,
                                                        mass_count_path=args.mass_count,
                                                        question_format_string=scorer_config['question_format_string'],
                                                        context=args.context)

        if qa_scorer is None:
            print(f'Loading model {scorer_config["model_name"]}...')
            qa_scorer = LMQuestionAnswerScorer.from_config(scorer_config)
        else:
            print(f'Updating prompting configuration for {scorer_config["model_name"]}...')
            qa_scorer.update_from_config(scorer_config)

        short_model_name = scorer_config["model_name"].split('/')[-1]
        bigrams_name = get_filename(args.bigrams).replace('_', '-')
        out_path = f'{args.out_dir}/predictions_{bigrams_name}_{short_model_name}_{scorer_config["config_nickname"]}.csv'
        dump_out_path = f'{args.out_dir}/dump_{scorer_config["config_nickname"]}.json'

        print(f'Getting predictions for {len(bigrams_with_questions)} questions...')
        model_responses = predict_model_responses(bigrams_with_questions, qa_scorer, dump_path=dump_out_path)

        print_metrics(model_responses)

        print(f'Writing full model responses to {out_path}')
        model_responses.to_csv(out_path, index=False)
