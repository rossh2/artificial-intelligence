import warnings
from collections import defaultdict
from typing import Union, Optional, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding

ACCELERATE_DEVICE_MAP_OPTIONS = ['auto', 'balanced', 'balanced_low_0', 'sequential']


class LMWrapper:
    """
    Credit to Kanishka Misra's minicons library
    Copied/adapted from IncrementalLMScorer
    """

    def __init__(self, model: Union[str, torch.nn.Module], tokenizer=None, padding_side: str = None,
                 system_prompt: str = '',
                 device: Optional[Union[str, int]] = 'cpu', precision: Optional[int] = None,
                 batch_size: int = 1, seed: int = None, hf_access_token: str = None, **kwargs):
        """
        :param model: should be path to a model (.pt or .bin file) stored
            locally, or name of a pretrained model stored on the Huggingface
            Model Hub, or a model (torch.nn.Module) that have the same
            signature as a Huggingface model obtained from
            `AutoModelForCausalLM`. In the last case, a corresponding tokenizer
            must also be provided.
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...} or "auto"` - auto will use Accelerate to map onto multiple GPUs
        :type device: str, optional
        :param tokenizer: if provided, use this tokenizer.
        :param kwargs: kwargs to pass to model
        """
        self.seed = seed
        if seed:
            # Use deterministic algorithms where possible but don't error if not possible
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.batch_size = batch_size
        self.hf_access_token = hf_access_token

        num_gpus = torch.cuda.device_count()
        if device in ACCELERATE_DEVICE_MAP_OPTIONS and num_gpus < 2:
            # Accelerate's device_map only applies for multiple GPUs
            if num_gpus == 1:
                device = 0
            else:
                device = "cpu"
            warnings.warn(f"Device map passed for accelerate but not enough GPUs present; "
                          f"setting device to {device}")
        self.device: Union[str, int] = device
        self.precision = precision

        self.system_prompt = system_prompt

        self.tokenizer = self.init_tokenizer(model, tokenizer, padding_side)
        self.padding_side = self.tokenizer.padding_side
        self.vocab = self.init_vocab()
        self.model = self.init_model(model, precision, **kwargs)

        self.setup_special_tokens(tokenizer)
        self.use_chat_template = self.tokenizer.chat_template is not None
        if self.use_chat_template:
            print('Using chat template provided by tokenizer')

        # n.B. No need to use accelerate to prepare the model if we're just doing inference

    def setup_special_tokens(self, tokenizer):
        # define CLS and SEP tokens
        if self.tokenizer.pad_token is None:
            if tokenizer is not None:
                warnings.warn(
                    "tokenizer is changed by adding pad_token_id to the tokenizer."
                )
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": ["<pad>"]}
                )
                self.tokenizer.pad_token = "<pad>"
                self.model.resize_token_embeddings(len(self.tokenizer))

    # noinspection PyMethodMayBeStatic
    def init_tokenizer(self, model, tokenizer, padding_side=None):
        if tokenizer is not None:
            if isinstance(tokenizer, str):
                if padding_side:
                    return AutoTokenizer.from_pretrained(tokenizer, padding_side=padding_side,
                                                         token=self.hf_access_token)
                else:
                    # Don't override padding_side with None if no padding side is passed
                    return AutoTokenizer.from_pretrained(tokenizer, token=self.hf_access_token)
            else:
                return tokenizer
        elif isinstance(model, str):
            if padding_side:
                return AutoTokenizer.from_pretrained(model, use_fast=True, padding_side=padding_side,
                                                     token=self.hf_access_token)
            else:
                # Don't override padding_side with None if no padding side is passed
                return AutoTokenizer.from_pretrained(model, use_fast=True, token=self.hf_access_token)
        else:
            raise Exception("Must provide either model name or tokenizer.")

    def init_vocab(self):
        vocab = defaultdict(list)

        for i in range(self.tokenizer.vocab_size):
            decoded = [(self.tokenizer.decode(i), i)]
            for x, j in decoded:
                vocab[x.strip()].append(j)

        return vocab

    def init_model(self, model: Union[str, torch.nn.Module], precision: int = None, **kwargs):
        if isinstance(model, str):
            if self.device in ACCELERATE_DEVICE_MAP_OPTIONS:
                print(f'Loading model {model} using accelerate with device map setting {self.device}')
                if precision:
                    if precision == 16:
                        # Set precision to 16-bit
                        loaded_model = AutoModelForCausalLM.from_pretrained(
                            model, device_map=self.device, torch_dtype=torch.float16, token=self.hf_access_token,
                            return_dict=True, **kwargs
                        )
                    elif precision == 8:
                        # Quantize to 8-bit
                        loaded_model = AutoModelForCausalLM.from_pretrained(
                            model, device_map=self.device, load_in_8bit=True, token=self.hf_access_token,
                            return_dict=True, **kwargs
                        )
                    else:
                        raise ValueError(f"Unsupported quantization: {precision}. Supported: 16 or 8-bit")
                else:
                    loaded_model = AutoModelForCausalLM.from_pretrained(
                        model, device_map=self.device, token=self.hf_access_token,
                        return_dict=True, **kwargs
                    )

                print("Device map:")
                print(loaded_model.hf_device_map)

            else:
                print(f'Loading model {model} on device {self.device}')
                if precision:
                    if precision == 16:
                        # Set precision to 16-bit
                        loaded_model = AutoModelForCausalLM.from_pretrained(
                            model, torch_dtype=torch.float16, token=self.hf_access_token,
                            return_dict=True, low_cpu_mem_usage=True, **kwargs
                        )
                    elif precision == 8:
                        loaded_model = AutoModelForCausalLM.from_pretrained(
                            model, load_in_8bit=True, token=self.hf_access_token,
                            return_dict=True, low_cpu_mem_usage=True, **kwargs
                        )
                    else:
                        raise ValueError(f"Unsupported quantization: {precision}. Supported: 16 or 8-bit")
                else:
                    loaded_model = AutoModelForCausalLM.from_pretrained(
                        model, return_dict=True, low_cpu_mem_usage=True, token=self.hf_access_token, **kwargs
                    )
                loaded_model.to(self.device)

            loaded_model.eval()

            return loaded_model
        else:
            return model

    def encode(self, text: Union[str, List[str]],
               bos_token: bool = False, eos_token: bool = False, padding: bool = True) -> BatchEncoding:

        def _format(self, text, bos, eos):
            if bos:
                text = self.tokenizer.bos_token + text
            if eos:
                text = text + self.tokenizer.eos_token
            return text

        text = [text] if isinstance(text, str) else text
        text = [_format(self, t, bos_token, eos_token) for t in text]
        encoded = self.tokenizer(text, return_tensors="pt", padding=padding)
        if "token_type_ids" in encoded.keys():
            encoded.pop("token_type_ids")

        return encoded
