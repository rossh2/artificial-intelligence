from typing import Union, List, Tuple, Callable, Iterable, Optional

import torch
from transformers import BatchEncoding

from lm_isa_experiment.hf_lm_wrapper import LMWrapper

ACCELERATE_DEVICE_MAP_OPTIONS = ['auto', 'balanced', 'balanced_low_0', 'sequential']


class LMContinuationScorer(LMWrapper):
    """
    Credit to Kanishka Misra's minicons library
    Copied/adapted from IncrementalLMScorer
    """

    def __init__(self, model: Union[str, torch.nn.Module], tokenizer=None,
                 system_prompt: str = '',
                 device: Optional[Union[str, int]] = 'cpu', precision: Optional[int] = None,
                 batch_size: int = 1, seed: int = None, hf_access_token=None, **kwargs):
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
        super().__init__(model=model, tokenizer=tokenizer,
                         system_prompt=system_prompt,
                         device=device, batch_size=batch_size, precision=precision, seed=seed,
                         hf_access_token=hf_access_token, **kwargs)

        if self.tokenizer.padding_side == "left":
            self.tokenizer.padding_side = "right"

        self.padding_side = self.tokenizer.padding_side

        # n.B. No need to use accelerate to prepare the model if we're just doing inference

    def prepare_text(self, text: Union[str, List[str], BatchEncoding],
                     bos_token: bool = False, eos_token: bool = False) -> Tuple:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on.

        :param text: batch of sentences to be prepared for scoring.

        :return: Batch of formatted input that can be passed to
            ``compute_stats``
        """
        if isinstance(text, BatchEncoding):
            encoded = text
        else:
            encoded = self.encode(text, bos_token, eos_token)
        offsets = [0] * len(encoded["input_ids"])
        return encoded, offsets

    def prime_text(self, preamble: Union[str, List[str]], stimuli: Union[str, List[str]],
                   separator=" ", bos_token=False, eos_token=False) -> Tuple:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on.

        :param ``Union[str, List[str]]`` preamble: Batch of prefixes/prime/preambles on which the LM is conditioned.
        :param ``Union[str, List[str]]`` stimuli: Batch of continuations that are scored based on the conditioned text
        (provided in the ``preamble``). The positions of the elements match their counterparts in the ``preamble``.

        :return: Batch of formatted input that can be passed to
            ``compute_stats``
        """
        preamble_text = [preamble] if isinstance(preamble, str) else preamble
        preamble_encoded = self.tokenizer(preamble_text)["input_ids"]
        preamble_lens = []
        for preamble_tokens in preamble_encoded:
            if bos_token:
                restricted_id = float("inf")
                bos_offset = 1
            else:
                restricted_id = self.tokenizer.pad_token_id
                bos_offset = 0
            preamble_lens.append(
                len([token for token in preamble_tokens if token != restricted_id])
                - 1
                + bos_offset
            )

        sentences = (
            [preamble + separator + stimuli]
            if isinstance(preamble, str)
            else [p + separator + s for p, s in list(zip(preamble, stimuli))]
        )

        return self.encode(sentences, bos_token, eos_token), preamble_lens

    def conditional_score(self, prefix: Union[str, List[str]], stimuli: Union[str, List[str]],
                          separator: str = " ",
                          reduction: Callable = lambda x: x.mean(0).item(),
                          prob: bool = False, base_two: bool = False, **kw) -> List[float]:
        """
        Pooled estimates of sequence log probabilities (or some modification of it), given a prefix. Pooling is usually done using a function that is passed to the method.

        :param prefix: a batch of prefixes or primes passed to the
            language model. This is what the sequence is conditioned on, and the model ignores the word probabilities of this part of the input in estimating the overall score.
        :type prefix: ``Union[str, List[str]]``
        :param stimuli: a batch of sequences (same length as prefix)
            that form the main input consisting of the sequence whose
            score you want to calculate.
        :type stimuli: ``Union[str, List[str]]``
        :param reduction: Reduction function, is selected to be
            ``lambda x: x.mean(0).item()`` by default, which stands for the avg. log-probability per token for each sequence in the batch.
        :type reduction: Callable
        :param ``bool`` prob: whether the model should return probabilities instead of log-probabilities. Can only be `True` when `base_two` is `False`.
        :param ``bool`` base_two: whether the base of the log should be 2 (usually preferred when reporting results in bits). Can only be `True` when `prob` is `False`.
        :param kw: model-specific keyword arguments to pass to the `prepare_text` function
        :return: List of floats specifying the desired score for the stimuli part of the input, e.g., P(stimuli | preamble).
        :rtype: ``List[float]``
        """
        primed = self.prime_text(prefix, stimuli, separator, **kw)

        result = self.compute_stats(
            primed, rank=False, base_two=base_two, prob=prob, return_tensors=True
        )
        logprob = result
        reduced = list(map(reduction, logprob))

        return reduced

    def compute_stats(
            self,
            batch: Iterable,
            rank: bool = False,
            prob: bool = False,
            base_two: bool = False,
            return_tensors: bool = False,
    ) -> Union[Tuple[List[float], List[float]], List[float]]:
        """
        Primary computational method that processes a batch of prepared sentences and returns per-token scores for each sentence. By default, returns log-probabilities.

        :param ``Iterable`` batch: batched input as processed by ``prepare_text`` or ``prime_text``.
        :param ``bool`` rank: whether the model should also return ranks per word (based on the conditional log-probability of the word in context).
        :param ``bool`` prob: whether the model should return probabilities instead of log-probabilities. Can only be `True` when `base_two` is `False`.
        :param ``bool`` base_two: whether the base of the log should be 2 (usually preferred when reporting results in bits). Can only be `True` when `prob` is `False`.
        :param ``bool`` return_tensors: whether the model should return scores as a list of tensors instead of a list of lists. This is important in some other convenient methods used in the package.

        :return: Either a tuple of lists, each containing probabilities and ranks per token in each sentence passed in the input.
        :rtype: ``Union[Tuple[List[float], List[int]], List[float]]``
        """
        assert not (
                base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        encoded, offsets = batch
        if self.device not in ACCELERATE_DEVICE_MAP_OPTIONS:
            encoded = encoded.to(self.device)

        # ids = [
        #     [i for i in instance if i != self.tokenizer.pad_token_id]
        #     for instance in encoded["input_ids"].tolist()
        # ]
        ids = [
            [i for i, am in zip(instance, attention_mask) if am != 0]
            for instance, attention_mask in zip(
                encoded["input_ids"].tolist(), encoded["attention_mask"].tolist()
            )
        ]

        ## Ignore the probabilities of the first token.
        effective_ids = [id[1:] for id in ids]

        with torch.no_grad():
            logits = self.model(**encoded).logits.detach()

        # logits[:, :, self.tokenizer.pad_token_id] = float("-inf")

        logits = logits.split([1] * len(offsets))

        ## Set up storage variables
        scores = []
        if rank:
            ranks = []

        for logit, idx, offset in zip(logits, effective_ids, offsets):
            length = len(idx)
            logit = logit.squeeze(0)[torch.arange(offset, length),]

            logprob_distribution = logit - logit.logsumexp(1).unsqueeze(1)
            query_ids = idx[offset:]
            if base_two:
                """
                Log_2(X) = log_e(X)/log_e(2) (broadcasted)
                """
                score = (
                        logprob_distribution[torch.arange(length - offset), query_ids]
                        / torch.tensor(2).log()
                ).tolist()
            else:
                if prob:
                    score = (
                        logprob_distribution[torch.arange(length - offset), query_ids]
                        .exp()
                        .tolist()
                    )
                else:
                    score = logprob_distribution[
                        torch.arange(length - offset), query_ids
                    ].tolist()

            if rank:
                # shape = logprob_distribution.shape
                """
                Double argsort trick:
                first argsort returns idxes of values that would return a sorted tensor,
                second argsort returns ranks (0 indexed)

                Proof: https://www.berkayantmen.com/rank.html

                TODO: Try to implement ranking in linear time but across arbitrary dimensions:
                https://stackoverflow.com/a/5284703
                """
                word_ranks = (-1.0 * logprob_distribution).argsort().argsort() + 1
                # inv_ranks = logprob_distribution.argsort().argsort() + 1
                # word_ranks = shape[1] - inv_ranks + 1
                word_ranks = word_ranks[
                    torch.arange(length - offset), query_ids
                ].tolist()
                ranks.append(word_ranks)

            scores.append(score)

        if return_tensors:
            scores = [torch.tensor(l) for l in scores]

        if rank:
            return scores, ranks
        else:
            return scores

    def sequence_score(
            self, batch, reduction=lambda x: x.mean(0).item(), base_two=False, **kwargs
    ):
        tokenized = self.prepare_text(batch, **kwargs)
        scores = self.compute_stats(
            tokenized, rank=False, base_two=base_two, return_tensors=True
        )
        reduced = list(map(reduction, scores))
        return reduced

    def token_score(
            self,
            batch: Union[str, List[str]],
            surprisal: bool = False,
            prob: bool = False,
            base_two: bool = False,
            rank: bool = False,
            decode: bool = True,
            **kwargs,
    ) -> Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]:
        """
        For every input sentence, returns a list of tuples in the following format:
            `(token, score)`,

        where score represents the log-probability (by default) of the token given context. Can also return ranks along with scores.

        :param ``Union[str, List[str]]`` batch: a single sentence or a batch of sentences.
        :param ``bool`` surprisal: If `True`, returns per-word surprisals instead of log-probabilities.
        :param ``bool`` prob: If `True`, returns per-word probabilities instead of log-probabilities.
        :param ``bool`` base_two: If `True`, uses log base 2 instead of natural-log (returns bits of values in case of surprisals)
        :param ``bool`` rank: If `True`, also returns the rank of each word in context (based on the log-probability value)

        :return: A `List` containing a `Tuple` consisting of the word, its associated score, and optionally, its rank.
        :rtype: ``Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]``
        """

        assert not (
                surprisal and prob
        ), "cannot both evaluate probability and surprisal at the same time!"
        assert not (
                base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        tokenized = self.prepare_text(batch, **kwargs)
        if rank:
            scores, ranks = self.compute_stats(
                tokenized, rank=rank, prob=prob, base_two=base_two, return_tensors=True
            )
        else:
            scores = self.compute_stats(
                tokenized, prob=prob, base_two=base_two, return_tensors=True
            )

        if surprisal:
            scores = [-1.0 * s for s in scores]

        scores = [s.tolist() for s in scores]

        indices = [
            [i for i, am in zip(instance, attention_mask) if am != 0]
            for instance, attention_mask in zip(
                tokenized[0]["input_ids"].tolist(),
                tokenized[0]["attention_mask"].tolist(),
            )
        ]
        if decode:
            tokens = [self.decode(idx) for idx in indices]
        else:
            tokens = [self.tokenizer.convert_ids_to_tokens(idx) for idx in indices]

        if rank:
            assert len(tokens) == len(scores) == len(ranks)
        else:
            assert len(tokens) == len(scores)

        res = []
        if rank:
            for t, s, r in zip(tokens, scores, ranks):
                if len(t) > len(s):
                    diff = len(t) - len(s)
                    sc = [0.0] * diff + s
                    ra = [0] * diff + r
                    res.append(list(zip(t, sc, ra)))
                else:
                    res.append(list(zip(t, sc, ra)))
        else:
            for t, s in zip(tokens, scores):
                if len(t) > len(s):
                    diff = len(t) - len(s)
                    sc = [0.0] * diff + s
                    res.append(list(zip(t, sc)))
                else:
                    res.append(list(zip(t, sc)))

        return res

    def decode(self, idx: List[int]):
        """
        Decode input ids using the model's tokenizer.

        :param ``List[int]`` idx: List of ids.

        :return: Decoded strings
        :rtype: List[str]
        """
        return [
            self.tokenizer.decode([x]).strip()
            for x in self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.convert_ids_to_tokens(idx)
            )
        ]
