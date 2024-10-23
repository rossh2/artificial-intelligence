import json
import re
from typing import Optional, Union, Dict, Any, List, Tuple, Iterable

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GenerationConfig

from lm_isa_experiment.hf_lm_wrapper import LMWrapper, ACCELERATE_DEVICE_MAP_OPTIONS


class ContextGenerationWrapper(LMWrapper):
    def __init__(self, model_name: str, device: Optional[Union[str, int]] = 'cpu',
                 seed: int = None, batch_size=1,
                 precision: int = None, hf_access_token: str = None,
                 max_tokens: int = 512, temperature: Optional[float] = 0.5,
                 response_count: int = 12,
                 response_template_string: str = '(.+)', triple_response_template_string='[123]. (.+)'):
        super().__init__(model=model_name, device=device, batch_size=batch_size,
                         precision=precision, seed=seed, hf_access_token=hf_access_token, padding_side='left')
        self.model_name = model_name

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.response_count = response_count
        self.response_template_string = response_template_string
        self.triple_response_template_string = triple_response_template_string

        self.generation_config = self.build_generation_config(max_tokens, temperature)

        if not self.use_chat_template:
            self.use_chat_template = True
            print('Warning: this model does not have a chat template; using the default chat template')

    def build_generation_config(self, max_tokens: int, temperature: float) -> GenerationConfig:
        gen_config = GenerationConfig.from_model_config(self.model.config)
        gen_config.pad_token_id = self.tokenizer.eos_token_id
        gen_config.max_new_tokens = max_tokens
        gen_config.do_sample = True
        gen_config.temperature = temperature
        return gen_config

    @staticmethod
    def from_config(config: Dict[str, Any]):
        gen_wrapper = ContextGenerationWrapper(model_name=(config.get('model_name')),
                                               device=config.get('device'),
                                               precision=config.get('precision'),
                                               seed=config.get('seed'),
                                               batch_size=config.get('batch_size'),
                                               hf_access_token=config.get('hf_access_token'),
                                               max_tokens=config.get('max_tokens'),
                                               temperature=config.get('temperature'),
                                               response_count=config.get('response_count'),
                                               response_template_string=config.get('response_template_string'),
                                               triple_response_template_string=config.get(
                                                   'triple_response_template_string')
                                               )
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
        self.temperature = config.get('temperature')
        self.response_count = config.get('response_count')

        self.generation_config = self.build_generation_config(self.max_tokens, self.temperature)

    def build_initial_messages(self, prompt_batch: List[str]) -> List[List[Dict[str, str]]]:
        messages_batch = []
        for prompt in prompt_batch:
            messages = []
            if self.system_prompt:
                messages.append({
                    "role": "system",
                    "content": self.system_prompt
                })
            messages.append({
                "role": "user",
                "content": prompt
            })
            messages_batch.append(messages)
        return messages_batch

    def build_subsequent_messages(self, previous_messages_batch: List[List[Dict[str, str]]],
                                  assistant_response_batch: List[str],
                                  prompt_batch: Iterable[str]) -> List[List[Dict[str, str]]]:
        messages_batch = []
        for prompt, previous_messages, assistant_response in zip(prompt_batch, previous_messages_batch,
                                                                 assistant_response_batch):
            new_messages = [
                {
                    "role": "assistant",
                    "content": assistant_response
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            messages_batch.append(previous_messages + new_messages)
        return messages_batch

    def generate_multiple_responses(self, prompts: List[Tuple[str, str, str]], dump_out_path: str = None) -> List[List[str]]:
        all_responses = []

        # noinspection PyTypeChecker
        # DataLoader takes many types, including any iterable such as our list
        dl = DataLoader(prompts, self.batch_size)

        for i, batch in enumerate(tqdm(dl, desc="Batches: ")):
            initial_prompt_batch, follow_up_prompt_batch, triple_prompt_batch = batch

            prompt_responses = [[] for _ in range(self.batch_size)]
            messages_batch = [[] for _ in range(self.batch_size)]
            last_model_response_batch = []

            while len(prompt_responses[0]) < self.response_count:
                if len(prompt_responses[0]) == 0:
                    prompt_to_use = initial_prompt_batch
                    template_to_use = self.response_template_string
                    expected_response_count = 1
                elif (self.response_count - len(prompt_responses[0])) % 3 == 0:
                    prompt_to_use = triple_prompt_batch
                    template_to_use = self.triple_response_template_string
                    expected_response_count = 3
                else:
                    prompt_to_use = follow_up_prompt_batch
                    template_to_use = self.response_template_string
                    expected_response_count = 1

                if len(messages_batch[0]) == 0:
                    messages_batch = self.build_initial_messages(initial_prompt_batch)
                else:
                    messages_batch = self.build_subsequent_messages(messages_batch, last_model_response_batch,
                                                                    prompt_to_use)

                text_input_batch = [self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                                       add_generation_prompt=True)
                                    for messages in messages_batch]
                if i == 0 and len(prompt_responses[0]) == 0:
                    print(f'Example prompt (previous responses: {len(prompt_responses)}):')
                    print(text_input_batch[0])
                model_inputs = self.encode_prompt(text_input_batch)

                output = self.model.generate(**model_inputs, generation_config=self.generation_config)
                responses = self.tokenizer.batch_decode(output, skip_special_tokens=False)
                last_model_response_batch = []
                for j, (response, text_input) in enumerate(zip(responses, text_input_batch)):
                    just_response = response.replace(text_input, '')
                    # Unclear if necessary to remove padding or if removed by batch decoding but remove just in case
                    last_model_response_batch.append(self.remove_padding(just_response))
                    cleaned_response = self.remove_special_tokens(just_response)
                    extracted_responses = re.findall(template_to_use, cleaned_response)
                    while len(extracted_responses) < expected_response_count:
                        # Model returned nothing, add empty strings so that we don't have an infinite loop
                        extracted_responses.append('')
                    prompt_responses[j].extend(extracted_responses)

            all_responses.extend(prompt_responses)
            if dump_out_path:
                with open(dump_out_path, 'a') as f:
                    f.write(json.dumps({
                        'batch_id': i,
                        'prompts': list(batch),
                        'responses': prompt_responses
                    }) + '\n')
        return all_responses

    def encode_prompt(self, prompt: Union[str, List[str]]):
        model_inputs = self.encode(prompt)
        if self.device not in ACCELERATE_DEVICE_MAP_OPTIONS:
            model_inputs = model_inputs.to(self.device)
        else:
            # If using auto, accelerate will handle copying the inputs to the
            # appropriate GPU. The documentation suggests moving the inputs to the first GPU anyway:
            # https://huggingface.co/docs/accelerate/v0.11.0/en/big_modeling
            # noinspection PyTypeChecker
            model_inputs = model_inputs.to(0)
        return model_inputs

    def remove_special_tokens(self, model_response: str):
        tokenized = self.tokenizer(model_response, add_special_tokens=False, padding=False)
        decoded = self.tokenizer.decode(tokenized['input_ids'], skip_special_tokens=True)
        return decoded

    def remove_padding(self, model_response: str):
        no_padding_response = model_response.replace(self.tokenizer.pad_token, '')
        return no_padding_response
