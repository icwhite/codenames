from typing import List, Optional, Tuple, Union

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama import Llama
from llama.generation import sample_top_p, Dialog, ChatPrediction, SPECIAL_TAGS, B_SYS, E_SYS, B_INST, E_INST, UNSAFE_ERROR

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer


LLAMA_7B_TEXT_CKPT_DIR = f'{MODELS_DIR}/llama-2-7b/'
LLAMA_7B_CHAT_CKPT_DIR = f'{MODELS_DIR}/llama-2-7b-chat/'
LLAMA_13B_TEXT_CKPT_DIR = f'{MODELS_DIR}/llama-2-13b/'
LLAMA_13B_CHAT_CKPT_DIR = f'{MODELS_DIR}/llama-2-13b-chat/'
LLAMA_TOKENIZER_PATH = f'{MODELS_DIR}/tokenizer/tokenizer.model'

def tile_seqs(seqs, n):
    """
    Taking in seqs=[a, b], n=3, will output [a, a, a, b, b, b]
    """
    output = [seq for seq in seqs for _ in range(n)]

    # sanity check
    assert set(output[:n]) == set([seqs[0]])

    return output

def repeat_seqs(seqs, n):
    """
    Taking in seqs=[a, b], n=3, will output [a, b, a, b, a, b]
    """
    output = seqs * n 

    # sanity check
    assert output[:len(seqs)] == seqs

    return output

class LlamaWrapper(Llama):
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        super().__init__(model, tokenizer)
    
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
        *args,
    ) -> "LlamaWrapper":
        """
        Build a LlamaWrapper instance by initializing and loading a pre-trained model.

        Differs from Llama.build in that itz returns a LlamaWrapper object instead of a Llama object.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            LlamaWrapper: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)
        else:
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)

        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return LlamaWrapper(model, tokenizer)
    
    def get_dialog_tokens(self, dialogs: List[Dialog]):
        """
        Used for chat model.
        """
        unsafe_requests = []
        prompt_tokens = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)
        return prompt_tokens, unsafe_requests

    def get_text_tokens(self, prompts: List[str], include_bos: bool = True, include_eos: bool = False):
        """
        Used for text model.
        """
        unsafe_requests = []
        prompt_tokens = []
        for prompt in prompts:
            unsafe_requests.append(
                any([tag in prompt for tag in SPECIAL_TAGS])
            )
            prompt_tokens.append(
                self.tokenizer.encode(prompt, bos=include_bos, eos=include_eos)
            )
        return prompt_tokens, unsafe_requests

    def get_fixed_output_prob(self,
        prompts: List[str],
        responses: List[str],
        temperature: float = 0.6,
    ) -> List[float]:
        """
        Gets probability of each (prompt with context, prompt without context, response) combination
        being generated under provided influence lambdas.

        Returns list of logprobs, ordered by (prompt 0, lambda 0), (prompt 0, lambda 1), ..., (prompt n, lambda m).
        E.g. the first logprob corresponds to the probability of generating the first response given the first prompt and first lambda.
        The last logprob corresponds to the probability of generating the last response given the last prompt and last lambda.
        """
        num_prompts, num_responses = len(prompts), len(responses)

        if len(prompts) != len(responses):
            prompts = tile_seqs(prompts, num_responses)
            responses = repeat_seqs(responses, num_prompts)

        assert len(prompts) == len(responses)

        # tokenize input strings. ignore unsafe requests output by text tokenizing function.
        prompt_tokens, _ = self.get_text_tokens(prompts)
        response_tokens, _ = self.get_text_tokens(responses, include_bos=False)

        probs = []
        batch_size = self.model.params.max_batch_size
        for i in range(0, len(prompt_tokens), batch_size):
            batch_prompt_toks = prompt_tokens[i: i + batch_size]
            batch_response_toks = response_tokens[i: i + batch_size]

            batch_probs = self.generate_fixed_output(
                prompt_tokens=batch_prompt_toks,
                response_tokens=batch_response_toks,
                temperature=temperature
            ) 
            probs.extend(batch_probs)

        # ordered by (p0, r0), (p0, r1), ... (pn, r0), (pn, r1), ... (pn, rm)
        return probs
    
    @torch.inference_mode()
    def generate_fixed_output(
        self,
        prompt_tokens: List[List[int]],
        response_tokens: List[List[int]],
        temperature: float = 0.6,
    ) -> List[float]:
        """        
        Calculates the probability of a supplied fixed response being generated.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts with context, where each prompt is represented as a list of integers.
            no_context_prompt_tokens (List[List[int]]): List of tokenized prompts without context, where each prompt is represented as a list of integers.
            response_tokens (List[List[int]]): List of tokenized fixed responses.
            infl_lambdas (List[float]): Lambda values for the influence function.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            
        Returns:
            List[float]: Cumulative log probabilities for each (lambda 0, prompt 0), (lambda 0, prompt 1), ..., (lambda m, prompt n) generation.
        """
        params = self.model.params
        bsz = len(prompt_tokens) # number of prompts
        
        pad_id = self.tokenizer.pad_id
        eos_id = self.tokenizer.eos_id

        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        assert len(prompt_tokens) == len(response_tokens)
        assert temperature > 0, "Temperature must be positive"

        # because we rely on the EOS token being generated, confirm that every response has an EOS at the end
        for resp in response_tokens:
            if resp[-1] != eos_id:
                resp.append(eos_id)
        assert all(resp[-1] == eos_id for resp in response_tokens)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        max_response_len = max(len(t) for t in response_tokens)

        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_prompt_len + max_response_len)

        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        resp_tokens = torch.full((bsz, max_response_len), pad_id, dtype=torch.long, device="cuda")

        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda") 

        for k, t in enumerate(response_tokens):
            resp_tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda") 
        
        resp_token_lens = torch.tensor([len(t) for t in response_tokens], device="cuda")

        # corresponds to (lambda 0, prompt 0), (lambda 0, prompt 1)
        fixed_generation_probs = torch.zeros((bsz,), dtype=torch.float, device="cuda")

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")

        # input_text_mask is true for prompt tokens
        input_text_mask = tokens != pad_id

        # Because every prompt may have a different length, we need to keep track of a different counter idx
        # for every prompt
        next_tok_counter = torch.zeros_like(eos_reached, dtype=resp_tokens.dtype, device="cuda")
        
        for cur_pos in range(min_prompt_len, total_len):
            # logits are of size (num prompts, sequence length, vocab size)
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            lps = F.log_softmax(logits[:, -1] / temperature, dim=-1)
            assert lps.size() == (bsz, self.model.params.vocab_size)

            # fix the next token to be from the previously generated response only when we haven't
            # hit fixed generation limits
            next_token = torch.where(
                next_tok_counter == resp_token_lens,
                torch.ones_like(next_tok_counter) * eos_id,
                torch.gather(resp_tokens, 1, next_tok_counter.unsqueeze(1)).squeeze(1)
            )
            assert next_token.shape == (bsz,)

            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            assert next_token.shape == (bsz,)

            # only update counter if prompt has been generated
            next_tok_counter += torch.where(
                # if the token is in the prompt, don't update the counter (add zero)
                input_text_mask[:, cur_pos], 0, 1
            )
            # if we've reached over the length of the prompt, continuously generate the last token
            next_tok_counter = torch.minimum(next_tok_counter, resp_token_lens - 1)

            # update eos vector before calculating next word probs
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )

            # save logprobs to output vector. mask out:
            # 1) generations w/ eos tokens
            # 2) generations w/ tokens that are already in the prompt
            # 3) exceed the length of generated tokens
            probs_mask = eos_reached | input_text_mask[:, cur_pos] | (next_tok_counter >= resp_token_lens)

            next_word_probs = torch.where(probs_mask, 
                                          torch.zeros_like(fixed_generation_probs), 
                                          lps.gather(1, next_token.unsqueeze(-1)).squeeze(1))
            
            assert fixed_generation_probs.size() == next_word_probs.size()
            fixed_generation_probs += next_word_probs

            tokens[:, cur_pos] = next_token
            
            prev_pos = cur_pos
            
            if all(eos_reached):
                break

        # sanity checks that we generated the correct tokens
        parse_until_eos = lambda toks: toks[:toks.index(eos_id)] if eos_id in toks else toks
        for (r, p, g) in zip(response_tokens, prompt_tokens, tokens.tolist()):
            resp = parse_until_eos(p + r)
            gen = parse_until_eos(g)
            assert resp == gen, f"Fixed response: {resp} != Generation: {gen}"

        del tokens
        del resp_tokens
        del eos_reached
        del next_tok_counter

        return fixed_generation_probs.tolist()
    
    def token_completion(
        self,
        prompts: Union[List[Dialog], List[str]],
        is_chat: bool = True,
        temperature: float = 0.6,
        top_n: int = 1,
        max_gen_len: Optional[int] = None,
        logprobs: bool = True,
    ) -> List[ChatPrediction]:
        """
        Generates the `top_n` next text or chat tokens for the provided `dialogs`, up to length `max_gen_len`.

        Args:
            prompts (List[Dialog] | List[str]): List of conversational dialog (where each dialog is a list of messages) or 
                a list of strings based on the value of `is_chat`.
            is_chat (bool, optional): Flag indicating whether to do chat or text completion. Defaults to True, i.e. chat completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling.
            top_n (int, optional): Top-n tokens to return.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.
        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        
        if is_chat:
            for p in prompts:
                assert isinstance(p, Dialog), "Expected list of dialogs"
            get_toks = self.get_dialog_tokens
        else:
            for p in prompts:
                assert isinstance(p, str), "Expected list of strings"
            get_toks = self.get_text_tokens
        prompt_tokens, unsafe_requests = get_toks(prompts)

        generation_tokens, generation_logprobs = self.generate_top_tokens(
            prompt_tokens=prompt_tokens,
            temperature=temperature,
            top_n=top_n,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]

    @torch.inference_mode()
    def generate_top_tokens(
        self,
        prompt_tokens: List[List[int]],
        temperature: float = 0.6,
        top_n: int = 1,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model, and return the top tokens and likelihoods

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        out_tokens, out_logprobs = [], []
        for idx in range(bsz):
            prompt_len = len(prompt_tokens[idx])
            assert prompt_len <= params.max_seq_len, f"prompt_len: {prompt_len}, params.max_seq_len: {params.max_seq_len}"
            total_len = min(params.max_seq_len, 1 + prompt_len)

            pad_id = self.tokenizer.pad_id
            tokens = torch.full((1, total_len), pad_id, dtype=torch.long, device="cuda")
            tokens[:, : len(prompt_tokens[idx])] = torch.tensor(prompt_tokens[idx], dtype=torch.long, device="cuda")
            if logprobs:
                token_logprobs = torch.zeros(top_n, dtype=torch.float)

            prev_pos = 0
            input_text_mask = tokens != pad_id
            assert temperature > 0
            assert prompt_len != total_len

            cur_pos = prompt_len
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos) # (nbatch, curr_len, ntokens)
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = return_top_n(probs, top_n)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)

            # tokens_ = tokens.repeat(top_n, 1)
            # logits_ = logits.repeat(top_n, 1, 1)

            # tokens_[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:] = torch.index_select(probs, 1, next_token).log()

            out_tokens.append([[val] for val in next_token.tolist()])
            if logprobs:
                out_logprobs.append([[val] for val in token_logprobs.tolist()])

        return (out_tokens, out_logprobs if logprobs else None)
    
def return_top_n(probs, n):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    next_token = probs_idx[:, :n]
    return next_token