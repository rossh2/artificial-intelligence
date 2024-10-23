from argparse import Namespace
from typing import Dict, Any


def update_scorer_config_from_args(scorer_config: Dict[str, Any], passed_args: Namespace):
    passed_args_dict = vars(passed_args)

    for arg in ['model_name', 'max_tokens', 'response_count', 'sample', 'temperature',
                'device', 'precision', 'batch_size', 'seed',
                'hf_access_token']:
        if arg in passed_args_dict:
            if scorer_config.get(arg, None):
                print(f'Overriding scorer config {arg} {scorer_config[arg]} with {passed_args_dict[arg]}')
            scorer_config[arg] = passed_args_dict[arg]
