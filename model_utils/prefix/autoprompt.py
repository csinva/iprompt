from typing import Optional, Tuple

import argparse
import random
import torch
import transformers

from .hotflip import HotFlip
from .utils import device, PrefixLoss, PrefixModel


class AutoPrompt(HotFlip):
    args: argparse.Namespace
    loss_func: PrefixLoss
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    prefix_ids: torch.Tensor
    prefix_embedding: torch.nn.Parameter
    preprefix: str
    def __init__(
            self,
            args: argparse.Namespace,
            loss_func: PrefixLoss,
            model: transformers.PreTrainedModel,
            tokenizer: transformers.PreTrainedTokenizer,
            preprefix: str = ''
        ):
        super().__init__(
            args=args, loss_func=loss_func, model=model, tokenizer=tokenizer, preprefix=preprefix
        )
        # AutoPrompt-specific parameters.
        self._num_candidates_per_prefix_token = args.hotflip_num_candidates # V_cand in autoprompt paper

    def compute_loss_and_call_backward(
            self,
            x_tokenized: transformers.BatchEncoding,
            y_tokenized: transformers.BatchEncoding,
            possible_answer_mask: torch.Tensor
        ) -> Tuple[torch.Tensor, int]:
        """Computes loss using `self.loss_func`.
        
        Returns:
            loss (float torch.Tensor) -- the loss
            num_correct (int): number of examples where prediction was correct
        """
        original_input_ids = x_tokenized.input_ids
        next_token_ids = y_tokenized.input_ids[:, 0] # only compute loss over next token

        _input_ids, loss, n_correct = self._compute_loss_with_set_prefix(
            original_input_ids=original_input_ids,
            next_token_ids=next_token_ids,
            possible_answer_mask=possible_answer_mask,
            prefix_ids=None,
        )
        loss.backward()

        # TODO check grad computation
        # TODO make sure grad is zero'd out between steps
        # 
        # Get top token replacements
        # 
        token_grads = self._prefix_token_grad
        assert token_grads.shape == (self._num_tokens, len(self.tokenizer.vocab))
        top_tokens_per_position = (
            token_grads.topk(k=self._num_candidates_per_prefix_token, dim=1, largest=False).indices
        )
        assert top_tokens_per_position.shape == (self._num_tokens, self._num_candidates_per_prefix_token)

        top_swap_tokens = top_tokens_per_position[self._swap_token_idx, :]
        #
        # Get most likely tokens.
        #
        top_swap_tokens = token_grads.argsort(descending=False).flatten()
        top_swap_tokens = top_swap_tokens[0:self._num_candidates_per_prefix_token]

        # rank candidates
        mask = torch.nn.functional.one_hot(
            torch.tensor(self._swap_token_idx), num_classes=self._num_tokens
        ).bool().to(device)
        candidate_prefix_ids = torch.where(mask, top_swap_tokens[:, None], self.prefix_ids[None, :])

        # get best prefix
        all_candidate_losses = torch.zeros(self._num_candidates_per_prefix_token, dtype=float).to(device)
        all_n_correct = torch.zeros(self._num_candidates_per_prefix_token, dtype=int).to(device)
        for i in range(self._num_candidates_per_prefix_token):
            with torch.no_grad():
                _cand_input_ids, cand_loss, cand_n_correct = (
                    self._compute_loss_with_set_prefix(
                        original_input_ids=original_input_ids,
                        next_token_ids=next_token_ids,
                        possible_answer_mask=possible_answer_mask,
                        prefix_ids=candidate_prefix_ids[i],
                    )
                )
            all_candidate_losses[i] += cand_loss
            all_n_correct[i] += cand_n_correct
        
        # compute new prefix
        new_prefix_idx = all_candidate_losses.argmin()
        new_prefix = candidate_prefix_ids[new_prefix_idx]
        new_prefix_str = self.tokenizer.decode(new_prefix)
        self._set_prefix_ids(new_prefix)
        new_prefix_loss = all_candidate_losses[new_prefix_idx]
        new_prefix_n_correct = all_n_correct[new_prefix_idx]
        print(f'New prefix: {new_prefix_str} Loss = {new_prefix_loss:.3f} n_correct={new_prefix_n_correct}')

        # this part is from the autoprompting codebase, even though
        # it's not mentioned in the paper
        self._swap_token_idx = random.randint(0, (self._num_tokens-1))

        return new_prefix_loss, new_prefix_n_correct
        
    def post_epoch(self, dataloader: torch.utils.data.DataLoader, possible_answer_mask: torch.Tensor) -> None:
        # 
        # Get candidate IDs for every position.
        # 
        pass