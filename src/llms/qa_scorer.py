import json
from operator import itemgetter
from typing import List, Tuple, Dict, Any, Union

from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from lm_isa_experiment.surprisal_scorer import LMContinuationScorer

EXAMPLE_QUESTIONS = [
    'Is a counterfeit scarf still a scarf?',
    'Is fake air still air?'
]


class LMQuestionAnswerScorer(LMContinuationScorer):
    YES_NO_ANSWERS = ['Yes', 'No']
    YES_NO_SUFFIX = ' Answer "Yes" or "No".'  # Includes space to separate from previous question

    def __init__(self, model_name: str = 'distilgpt2',
                 device: Union[str, int] = 'cpu', batch_size: int = 16,
                 precision: int = None, seed: int = None, hf_access_token: str = None):
        super().__init__(model=model_name, device=device, batch_size=batch_size, precision=precision,
                         seed=seed, hf_access_token=hf_access_token)
        self.model_name = model_name

        self.responses = []
        self.response_sep = ' '
        self.few_shot_prefix: Union[str, List[Dict[str, str]]] = [] if self.use_chat_template else ''
        self.prompt_instruction_prefix = ''
        self.prompt_instruction_suffix = ''

    def set_responses(self, responses: List[str], response_separator: str = ' ',
                      prompt_instruction_prefix: str = '', prompt_instruction_suffix: str = ''):
        self.responses = responses
        self.response_sep = response_separator
        self.prompt_instruction_prefix = prompt_instruction_prefix
        self.prompt_instruction_suffix = prompt_instruction_suffix

    def set_few_shot_prefix(self, few_shot_prompts: List[str], few_shot_answers: List[str], few_shot_sep='\n\n'):
        if self.use_chat_template:
            messages = []
            if self.system_prompt:
                messages.append({
                    "role": "system",
                    "content": self.system_prompt
                })
            for prompt, answer in zip(few_shot_prompts, few_shot_answers):
                full_prompt = self.build_single_prompt(prompt)
                messages.append({
                    "role": "user",
                    "content": full_prompt
                })
                messages.append({
                    "role": "assistant",
                    "content": answer
                })

            self.few_shot_prefix = messages
        else:
            examples = []
            for prompt, answer in zip(few_shot_prompts, few_shot_answers):
                full_prompt = self.build_single_prompt(prompt)
                example = full_prompt + self.response_sep + answer
                examples.append(example)

            self.few_shot_prefix = few_shot_sep.join(examples)
            if self.few_shot_prefix:
                self.few_shot_prefix = self.few_shot_prefix + few_shot_sep

    @staticmethod
    def from_config(config: Dict[str, Any]):
        qa_scorer = LMQuestionAnswerScorer(model_name=(config.get('model_name')),
                                           device=config.get('device'),
                                           batch_size=config.get('batch_size'),
                                           precision=config.get('precision'),
                                           seed=config.get('seed'),
                                           hf_access_token=config.get('hf_access_token'))
        qa_scorer.set_responses(responses=config.get('responses'),
                                response_separator=config.get('response_separator'),
                                prompt_instruction_prefix=config.get('prompt_instruction_prefix'),
                                prompt_instruction_suffix=config.get('prompt_instruction_suffix'))
        qa_scorer.set_few_shot_prefix(few_shot_prompts=config.get('few_shot_prompts'),
                                      few_shot_answers=config.get('few_shot_answers'),
                                      few_shot_sep=config.get('few_shot_sep'))
        return qa_scorer

    def update_from_config(self, config: Dict[str, Any]):
        if (config.get('model_name') != self.model_name or config.get('device') != self.device
                or config.get('precision') != self.precision or config.get('seed') != self.seed
                or config.get('hf_access_token') != self.hf_access_token):
            raise ValueError('Cannot update model, device, precision, seed or access token. '
                             'You must create a new instance of the class')

        self.batch_size = config.get('batch_size')

        self.set_responses(responses=config.get('responses'),
                           response_separator=config.get('response_separator'),
                           prompt_instruction_prefix=config.get('prompt_instruction_prefix'),
                           prompt_instruction_suffix=config.get('prompt_instruction_suffix'))
        self.set_few_shot_prefix(few_shot_prompts=config.get('few_shot_prompts'),
                                 few_shot_answers=config.get('few_shot_answers'),
                                 few_shot_sep=config.get('few_shot_sep'))

    def build_single_prompt(self, prompt):
        return self.prompt_instruction_prefix + prompt + self.prompt_instruction_suffix

    def build_full_prompt(self, prompt):
        target_prompt = self.build_single_prompt(prompt)
        if self.use_chat_template:
            messages = self.few_shot_prefix + [{
                "role": "user",
                "content": target_prompt
            }]
            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            full_prompt = self.few_shot_prefix + target_prompt
        return full_prompt

    def build_prompt_response_pairs(self, prompts: List[str]) -> DataLoader:
        if not self.responses:
            raise ValueError('Responses not set')

        print('Building prompts...')
        full_prompts = [self.build_full_prompt(prompt) for prompt in prompts]

        pairs = [(prompt, response) for prompt in full_prompts for response in self.responses]

        print('Example prompt:')
        print(f'{pairs[0][0]}{self.response_sep}{pairs[0][1]}')

        # noinspection PyTypeChecker
        # DataLoader takes many types, including any iterable such as our list
        return DataLoader(pairs, self.batch_size)

    def get_response_surprisals(self, prompts: List[str], dump_out_path: str = None) -> List[List[Tuple[str, float]]]:
        """
        :param prompts: a list of prompts
        :param dump_out_path: file to write to while iterating
        :return: (a list containing) for each prompt, a list containing the log probability for each response
        (as a tuple of response followed by surprisal)
        """

        # noinspection PyTypeChecker
        # DataLoader takes many times, including any iterable such as our list
        prompt_response_dl = self.build_prompt_response_pairs(prompts)

        prompts = []
        new_prompt = True
        results_by_prompt = [[]]
        predictions = []
        flat_results = []
        for batch in tqdm(prompt_response_dl, desc="Batches: "):
            prompt_batch, response_batch = batch
            scores = self.conditional_score(prompt_batch, response_batch, separator=self.response_sep)
            surprisals = [-1.0 * s for s in scores]
            response_surprisals = list(zip(response_batch, surprisals))

            # Group results by response to return one list item per prompt input
            prompt_batch = list(prompt_batch)
            while len(response_surprisals) > 0:
                while len(results_by_prompt[-1]) < len(self.responses) and len(response_surprisals) > 0:
                    prompt = prompt_batch.pop(0)
                    if new_prompt:
                        prompts.append(prompt)
                        new_prompt = False
                    results_by_prompt[-1].append(response_surprisals.pop(0))
                if len(response_surprisals) > 0:
                    # We finished one prompt, moving on to the next
                    new_prompt = True
                    # Prediction is the item with least surprisal
                    predicted_response = self.get_predicted_response(results_by_prompt[-1])
                    predictions.append(predicted_response)
                    results_by_prompt.append([])
            flat_results.extend(list(zip(response_batch, surprisals)))

            if dump_out_path:
                with open(dump_out_path, 'w') as f:
                    # Overwrite previous dump with new values
                    f.write(json.dumps({
                        'prompts': prompts,
                        'results_by_prompt': results_by_prompt,
                        'predictions': predictions
                    }) + '\n')

        return results_by_prompt

    def get_predicted_responses(self, surprisals_by_response: List[List[Tuple[str, float]]]):
        predictions = []
        for prompt_results in surprisals_by_response:
            predicted_response = self.get_predicted_response(prompt_results)
            predictions.append(predicted_response)
        return predictions

    @staticmethod
    def get_predicted_response(prompt_results: List[Tuple[str, float]]) -> str:
        # Prediction is the item with least surprisal
        predicted_response = min(prompt_results, key=itemgetter(1))[0]
        return predicted_response

    def calculate_metrics_from_surprisal(self, surprisals_by_response: List[List[Tuple[str, float]]],
                                         desired_answers: List[str]) -> Dict[str, float]:
        predictions = self.get_predicted_responses(surprisals_by_response)
        f1 = f1_score(y_pred=predictions, y_true=desired_answers, average='macro')
        acc = accuracy_score(y_pred=predictions, y_true=desired_answers)
        return {
            'f1': f1,
            'accuracy': acc
        }

    def evaluate_model_surprisal(self, prompts: List[str], desired_answers: List[str]) -> Dict[str, float]:
        surprisals_by_response = self.get_response_surprisals(prompts)
        return self.calculate_metrics_from_surprisal(surprisals_by_response, desired_answers)


if __name__ == '__main__':
    test_scorer = LMQuestionAnswerScorer(model_name='facebook/opt-350m')
    test_scorer.set_responses(test_scorer.YES_NO_ANSWERS, prompt_instruction_suffix=test_scorer.YES_NO_SUFFIX)
    example_results = test_scorer.get_response_surprisals(EXAMPLE_QUESTIONS)

    for q, q_results in zip(EXAMPLE_QUESTIONS, example_results):
        print(q)
        for rsp, rsp_srp in q_results:
            print(f'{rsp}: {rsp_srp:.3f}')

    metrics = test_scorer.calculate_metrics_from_surprisal(example_results, ['Yes', 'No'])
    print('Overall metrics:')
    for metric, value in metrics.items():
        print(f'{metric}: {value:.2f}')
