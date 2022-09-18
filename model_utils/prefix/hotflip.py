import argparse

import torch
import torch.nn as nn
import transformers

from .utils import PrefixModel


VERBOSE = False # whether to print grads, etc.
TOP_K = 20 # for printing grads, etc.

class HotFlipPrefixModel(PrefixModel):
    args: argparse.Namespace
    loss_func: PrefixLoss
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    prefix_ids: torch.Tensor
    prefix_embedding: nn.Parameter
    preprefix: str
    def __init__(self, args: argparse.Namespace, loss_func: PrefixLoss, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer,
        preprefix: str = 'The function to compute is'):
        super().__init__(
            args=args, loss_func=loss_func, model=model, tokenizer=tokenizer, preprefix=preprefix
        )
        # HotFlip-specific parameters.
        self._min_loss = float('inf')
        self._num_tokens = args.num_learned_tokens # TODO argparse for n_tokens
        self._num_candidates_per_prefix_token = args.hotflip_num_candidates # TODO argparse for this too
        self._swap_token_idx = 0
        # Sort both a version with a preprefix ("The function to compute is") and a version
        # where the full prefix is discovered by HotFlip without any assistance.
        if preprefix:
            self.preprefix_ids = self.tokenizer.encode(preprefix, return_tensors='pt')
        else:
            self.preprefix_ids = None
        self.prefix_ids = self.init_discrete_prefix(num_tokens=self._num_tokens)
        self.prefix_embedding = nn.Parameter(
            self.token_embedding.forward(self.prefix_ids), requires_grad=True
        )
        print(f"preprefix: '{preprefix}'")

        # disable grads to model
        for p in self.model.parameters(): p.requires_grad = False
    
    def _set_prefix_ids(self, new_ids: torch.Tensor) -> None:
        self.prefix_ids = new_ids
        # self.prefix_embedding = self.init_continuous_prefix()
        self.prefix_embedding = nn.Parameter(
            self.token_embedding.forward(self.prefix_ids), requires_grad=True
        )

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
            possible_answer_mask=possible_answer_mask
        )

        # self._set_prefix_ids(best_prefix)
        return loss, n_correct
        
    def post_epoch(self, dataloader: torch.utils.data.DataLoader, possible_answer_mask: torch.Tensor) -> None:
        # 
        # Get candidate IDs for every position.
        # 
        token_idx = self._swap_token_idx
        token_grads = self._prefix_token_grad
        top_tokens_per_position = (
            token_grads.topk(k=self._num_candidates_per_prefix_token, dim=1, largest=False).indices
        )
        # breakpoint()
        assert top_tokens_per_position.shape == (self._num_tokens, self._num_candidates_per_prefix_token)

        top_swap_tokens = top_tokens_per_position[token_idx, :]
        #
        # Get most likely tokens.
        #
        # prefix_until_swap_ids = torch.cat(
        #     (self.preprefix_ids.to(device), self.prefix_ids[None, :token_idx].to(device)), dim=1
        # ).to(device)
        swap_token_logits = self.model(self.preprefix_ids.to(device)).logits[:, -1, :]

        rvocab = {v: k for k,v in self.tokenizer.vocab.items()}
        # dist_sum = (swap_token_logits.log_softmax(dim=1) * .7 + (-1 * token_grads).log_softmax(dim=1))
        # for v in (swap_token_logits.log_softmax(dim=1) * .7 + (-1 * token_grads).log_softmax(dim=1)).topk(10).indices.flatten(): print(rvocab[v.item()])


        token_losses = (
            (swap_token_logits.log_softmax(dim=1) * 1.0 + (-1 * token_grads).log_softmax(dim=1))
        )
        top_swap_tokens = token_losses.argsort(descending=True).flatten()
        top_swap_tokens = top_swap_tokens[:self._num_candidates_per_prefix_token]
        # 
        # Evaluate candidates.
        # 
        all_candidate_losses = torch.zeros(self._num_candidates_per_prefix_token, dtype=float).to(device)
        all_n_correct = torch.zeros(self._num_candidates_per_prefix_token, dtype=int).to(device)
        best_loss = self._min_loss

        mask = torch.nn.functional.one_hot(
            torch.tensor(token_idx), num_classes=self._num_tokens
        ).bool().to(device)

        for batch in tqdm.tqdm(dataloader, desc='evaluating HotFlip candidates', colour='red', leave=False):
            # Loop in this order so we only tokenize each thing once.
            x_text = [f'. {prompt}' for prompt in batch['input']]
            y_text = [answer.replace('.', '').rstrip() for answer in batch['output']] # strip newlines and periods.
            #
            input_ids = self.tokenizer(x_text, return_tensors='pt', padding='longest')['input_ids'].to(device)
            next_token_ids = self.tokenizer(y_text, return_tensors='pt', padding='longest')['input_ids'].to(device)
            # only evaluate on single next-token
            next_token_ids = next_token_ids[:, 0]
            for candidate_idx in range(self._num_candidates_per_prefix_token):
                new_token_id = top_swap_tokens[candidate_idx]
                prefix_ids = torch.where(
                    mask, new_token_id, self.prefix_ids.to(device)
                ).to(device)
                with torch.no_grad():
                    _input_ids, loss, n_correct = (
                        self._compute_loss_with_set_prefix(
                            original_input_ids=input_ids,
                            next_token_ids=next_token_ids,
                            possible_answer_mask=possible_answer_mask,
                            prefix_ids=prefix_ids
                        )
                    )
                all_candidate_losses[candidate_idx] += loss
                all_n_correct[candidate_idx] += n_correct

        #
        # Recreate prefix with min loss.
        #
        min_loss = all_candidate_losses.min()
        best_candidate_idx = all_candidate_losses.argmin()

        new_token_id = top_swap_tokens[best_candidate_idx]
        best_prefix_ids = torch.where(
            mask, new_token_id, self.prefix_ids.to(device)
        ).to(device)

        # if loss < self._min_loss:
        #     self._min_loss = loss
        #     best_prefix_ids = prefix_ids

        # 
        # Pick top candidate and reset self._min_loss. (TODO: Support beam width > 1.)
        # 
        print(f'[Loss = {min_loss/len(dataloader):.2f}] // Old prefix: {self.tokenizer.decode(self.prefix_ids)} // New prefix: {self.tokenizer.decode(best_prefix_ids)} // New n_correct = {all_n_correct[best_candidate_idx].item()}')

        if (best_prefix_ids == self.prefix_ids.to(device)).all():
            print('no change in prefix, exiting')
            exit()
        self._set_prefix_ids(best_prefix_ids)
        # breakpoint()

        self._swap_token_idx = (self._swap_token_idx + 1) % self._num_tokens
        return

    @property
    def prefix_embedding_token_ids(self) -> torch.Tensor:
        return self.prefix_embedding.argmax(dim=-1)

    @property
    def trainable_params(self) -> Iterable[nn.Parameter]:
        return [self.prefix_embedding]

    def embed_input_ids(self, input_ids: torch.Tensor, prefix_ids: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets token embeddings for tokens given by `input_ids` prefixed by `prefix_ids`.

        If not provided, `prefix_ids` is replaced with `self.prefix_ids`
        at every position.

        Args:
            input_ids (int torch.Tensor) -- IDs for batch of sentences
            prefix_ids (Optional int torch.Tensor) -- IDs for a single prefix
                to be prepended before each input ID. If not provided,
                will be overridden with prefix from `self.prefix_ids`.

        Returns:
            input_ids (int torch.Tensor) -- IDs of all tokens, including prefix
            outputs (float torch.Tensor): embedded tokens
        """
        batch_size = len(input_ids)
        if prefix_ids is None:
            prefix_ids = self.prefix_ids
            prefix_embedding = self.prefix_embedding
            
        else:
            prefix_embedding = self.token_embedding.forward(prefix_ids)

        # Support an optional preprefix to help guide the learned prefix.
        prefix_ids = prefix_ids.to(device).repeat((batch_size, 1))
        if self.preprefix_ids is None:
            full_input_ids = torch.cat(
                (prefix_ids, input_ids), dim=1
            )
            outputs = torch.cat(
                (
                    prefix_embedding[None].repeat((batch_size, 1, 1)),
                    self.token_embedding.forward(input_ids)
                ), dim=1
            )
        else:
            # concatenate preprefix (fixed) + prefix (learned) + example
            preprefix_ids = self.preprefix_ids.to(device).repeat((batch_size, 1))
            full_input_ids = torch.cat(
                (preprefix_ids, prefix_ids, input_ids), dim=1
            )
            outputs = torch.cat(
                (
                    self.token_embedding.forward(preprefix_ids),
                    prefix_embedding[None].repeat((batch_size, 1, 1)),
                    self.token_embedding.forward(input_ids)
                ), dim=1
            )
        return full_input_ids, outputs