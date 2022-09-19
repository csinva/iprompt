import argparse
import torch
import transformers

from .hotflip import HotFlip
from .utils import PrefixLoss, PrefixModel


class AutoPrompt(HotFlip):
    args: argparse.Namespace
    loss_func: PrefixLoss
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    prefix_ids: torch.Tensor
    prefix_embedding: torch.nn.Parameter
    preprefix: str
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # AutoPrompt-specific parameters.
        self._num_candidates_per_prefix_token = args.hotflip_num_candidates # V_cand in autoprompt paper
    
    def _set_prefix_ids(self, new_ids: torch.Tensor) -> None:
        self.prefix_ids = new_ids.to(device)
        self.prefix_embedding = torch.nn.Parameter(
            self.token_embedding.to(device).forward(self.prefix_ids), requires_grad=True
        )
        # track prefixes we've tried
        self._tested_prefix_ids[(tuple(new_ids.flatten().tolist()), self._swap_token_idx)] += 1

    def pre_epoch(self) -> None:
        # Print closest tokens at the beginning of each epoch.
        if VERBOSE:
            print("*" *  30)
            print(f"Epoch {epoch}. Closest tokens to '{prefix_str}':")
            word_distances =  ((self.token_embedding.weight - self.prefix_embedding.reshape(1, emb_dim))**2).sum(1)
            assert word_distances.shape == (50_257,)
            topk_closest_words = distances = word_distances.topk(k=TOP_K, largest=False)
            for _id, _dist in zip(topk_closest_words.indices.cpu().tolist(), topk_closest_words.values.cpu().tolist()):
                print(f'\t{self.id_to_word[_id]} ({_id}): {_dist:.3f}')
            print("*" * 30)
    
    @property
    def _prefix_token_grad(self) -> torch.Tensor:
        """Gradient of the prefix tokens wrt the token embedding matrix."""
        return torch.einsum('nd,vd->nv', self.prefix_embedding.grad, self.token_embedding.weight)

    def _compute_loss_with_set_prefix(
            self,
            original_input_ids: torch.Tensor,
            next_token_ids: torch.Tensor,
            possible_answer_mask: torch.Tensor,
            prefix_ids: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        input_ids, outputs = self.forward(input_ids=original_input_ids, prefix_ids=prefix_ids)

        next_token_logits = outputs.logits[:, -1, :]
        n_correct = (
            next_token_logits.argmax(dim=-1)
                ==
            next_token_ids
        ).int().sum()

        original_loss = self.loss_func(
            input_ids=input_ids,
            next_token_ids=next_token_ids,
            logits=outputs['logits'],
            answer_mask=possible_answer_mask
        )
        return input_ids, original_loss, n_correct

    def compute_loss(
            self,
            original_input_ids: torch.Tensor,
            next_token_ids: torch.Tensor,
            possible_answer_mask: torch.Tensor
        ) -> Tuple[torch.Tensor, int]:
        """Computes loss using `self.loss_func`.
        
        Returns:
            loss (float torch.Tensor) -- the loss
            num_correct (int): number of examples where prediction was correct
        """
        _input_ids, loss, n_correct = self._compute_loss_with_set_prefix(
            original_input_ids=original_input_ids,
            next_token_ids=next_token_ids,
            possible_answer_mask=possible_answer_mask,
            prefix_ids=None,
        )


        # TODO check grad computation
        # TODO make sure grad is zero'd out between steps
        # 
        # Get top token replacements
        # 
        token_grads = self._prefix_token_grad
        top_tokens_per_position = (
            token_grads.topk(k=self._num_candidates_per_prefix_token, dim=1, largest=False).indices
        )
        assert top_tokens_per_position.shape == (self._num_tokens, self._num_candidates_per_prefix_token)

        top_swap_tokens = top_tokens_per_position[token_idx, :]
        #
        # Get most likely tokens.
        #
        top_swap_tokens = token_losses.argsort(descending=False).flatten()
        top_swap_tokens = top_swap_tokens[0:self._num_candidates_per_prefix_token]

        # rank candidates
        mask = torch.nn.functional.one_hot(
            torch.tensor(self._swap_token_idx), num_classes=self._num_tokens
        ).bool().to(device)
        candidate_prefix_ids = torch.where(
            mask[None], top_swap_tokens[None], self.prefix_ids
        )

        # get best prefix
        all_candidate_losses = torch.zeros(self._num_candidates_per_prefix_token, dtype=float).to(device)
        all_n_correct = torch.zeros(self._num_candidates_per_prefix_token, dtype=int).to(device)
        for i in range(self._num_candidates_per_prefix_token):
            with torch.no_grad():
                best_prefix = _cand_input_ids, cand_loss, cand_n_correct = (
                    self._compute_loss_with_set_prefix(
                        original_input_ids=original_input_ids,
                        next_token_ids=next_token_ids,
                        possible_answer_mask=possible_answer_mask,
                        prefix_ids=None,
                    )
                )
            all_candidate_losses += cand_loss
            all_n_correct += cand_n_correct
        
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

        return loss, n_correct
        
    def post_epoch(self, dataloader: torch.utils.data.DataLoader, possible_answer_mask: torch.Tensor) -> None:
        # 
        # Get candidate IDs for every position.
        # 
        pass