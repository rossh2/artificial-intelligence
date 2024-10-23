import json
from typing import List, Tuple, Dict, Any, Optional, Union

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GenerationConfig

from lm_isa_experiment.hf_lm_wrapper import LMWrapper, ACCELERATE_DEVICE_MAP_OPTIONS


class LMGenerationWrapper(LMWrapper):
    def __init__(self, model_name: str, device: Optional[Union[str, int]] = 'cpu',
                 batch_size: int = 1, seed: int = None,
                 precision: int = None, hf_access_token: str = None,
                 prompt_prefix: str = '', prompt_suffix: str = '', response_separator: str = ' ',
                 max_tokens: int = 200, sample: bool = False):
        super().__init__(model=model_name, device=device, batch_size=batch_size,
                         precision=precision, seed=seed, hf_access_token=hf_access_token, padding_side='left')
        self.model_name = model_name

        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.response_sep = response_separator
        self.few_shot_prefix: Union[str, List[Dict[str, str]]] = [] if self.use_chat_template else ''
        self.max_tokens = max_tokens
        self.sample = sample

        self.generation_config = self.build_generation_config(max_tokens, sample)

    @staticmethod
    def from_config(config: Dict[str, Any]):
        gen_wrapper = LMGenerationWrapper(model_name=(config.get('model_name')),
                                          device=config.get('device'),
                                          batch_size=config.get('batch_size'),
                                          precision=config.get('precision'),
                                          seed=config.get('seed'),
                                          hf_access_token=config.get('hf_access_token'),
                                          response_separator=config.get('response_separator'),
                                          prompt_prefix=config.get('prompt_instruction_prefix'),
                                          prompt_suffix=config.get('prompt_instruction_suffix'),
                                          max_tokens=config.get('max_tokens'),
                                          sample=config.get('sample')
                                          )
        gen_wrapper.set_few_shot_prefix(few_shot_prompts=config.get('few_shot_prompts'),
                                        few_shot_answers=config.get('few_shot_answers'),
                                        few_shot_sep=config.get('few_shot_sep'))
        return gen_wrapper

    def update_from_config(self, config: Dict[str, Any]):
        if (config.get('model_name') != self.model_name or config.get('device') != self.device
                or config.get('precision') != self.precision or config.get('seed') != self.seed
                or config.get('hf_access_token') != self.hf_access_token):
            raise ValueError('Cannot update model, device, precision, seed or access token. '
                             'You must create a new instance of the class')
        if config.get('batch_size') != self.batch_size:
            print('Warning: updating batch size. Generation at different batch sizes with otherwise identical settings'
                  'will result in different results.')
        self.batch_size = config.get('batch_size')

        self.max_tokens = config.get('max_tokens')
        self.sample = config.get('sample')

        self.response_sep = config.get('response_separator')
        self.prompt_prefix = config.get('prompt_prefix')
        self.prompt_suffix = config.get('prompt_suffix')

        self.set_few_shot_prefix(few_shot_prompts=config.get('few_shot_prompts'),
                                 few_shot_answers=config.get('few_shot_answers'),
                                 few_shot_sep=config.get('few_shot_sep'))

        self.generation_config = self.build_generation_config(self.max_tokens, self.sample)

    # TODO redo the class hierarchy so that this method can be shared between generation and surprisal
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
                    "content": "answer"
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

    def build_generation_config(self, max_tokens: int, sample: bool) -> GenerationConfig:
        gen_config = GenerationConfig.from_model_config(self.model.config)
        gen_config.pad_token_id = self.tokenizer.eos_token_id
        gen_config.max_new_tokens = max_tokens
        gen_config.do_sample = sample
        return gen_config

    def build_single_prompt(self, prompt):
        return self.prompt_prefix + prompt + self.prompt_suffix

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

    def generate_responses(self, prompts: List[str]) -> List[Tuple[str, str]]:
        print('Building prompts...')
        prompts = [self.build_full_prompt(prompt) for prompt in prompts]

        print('Example prompt:')
        print(prompts[0])

        # noinspection PyTypeChecker
        # DataLoader takes many types, including any iterable such as our list
        dl = DataLoader(prompts, self.batch_size)

        prompt_responses = []
        for batch in tqdm(dl, desc='Batches: '):
            model_inputs = self.encode(batch)

            if self.device not in ACCELERATE_DEVICE_MAP_OPTIONS:
                model_inputs = model_inputs.to(self.device)
            else:
                # If using auto, accelerate will handle copying the inputs to the
                # appropriate GPU. The documentation suggests moving the inputs to the first GPU anyway:
                # https://huggingface.co/docs/accelerate/v0.11.0/en/big_modeling
                # noinspection PyTypeChecker
                model_inputs = model_inputs.to(0)

            output = self.model.generate(**model_inputs, generation_config=self.generation_config)
            responses = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            prompt_responses.extend(list(zip(prompts, responses)))

        return prompt_responses


if __name__ == '__main__':
    with open('lm_isa_experiment/config/isa_yesno_2shot_scoring_config.json') as config_file:
        scorer_config = json.load(config_file)
    scorer_config['model_name'] = 'distilgpt2'
    scorer_config['batch_size'] = 8
    scorer_config['device'] = 'cpu'
    scorer_config['max_tokens'] = 200
    wrapper = LMGenerationWrapper.from_config(scorer_config)

    sample_prompts = ['is a fake door still a door?']
    print('Generating responses')
    prompt_response = wrapper.generate_responses(sample_prompts)

    print(prompt_response)
