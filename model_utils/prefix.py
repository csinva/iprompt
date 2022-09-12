from typing import Any, Dict, Iterable, Optional, Tuple

import abc
import dataclasses
import functools
import random
import transformers
import torch
from torch import nn
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclasses.dataclass
class PrefixLoss:
    """Computes next-token-prediction loss with optional language modeling component.
    """
    gamma: float
    tokenizer: transformers.PreTrainedTokenizer # for debugging

    def _compute_fluency_loss(
            self, logits: torch.Tensor, input_ids: torch.Tensor
        ) -> torch.Tensor:
        if self.gamma == 0:
            return torch.tensor(0.0).to(device)
        # logits gives us the probability of each token that comes after each token in input_ids.
        # so they have the same shape. But we only want to compute ppl using the tokens we have,
        # i.e. not the first true token (which we don't have logits for) or the last predicted token
        # (which we don't know the true id for). so we have to shift each by one index.
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]

        # now flatten along sequence length so we can compute crossentropy.
        batch_size, sequence_length, vocab_size = logits.shape
        assert input_ids.shape == (batch_size, sequence_length)
        logits = logits.reshape((batch_size * sequence_length, vocab_size))
        input_ids = input_ids.reshape((batch_size * sequence_length, ))
        
        loss = torch.nn.functional.cross_entropy(
            input=logits,
            target=input_ids,
            reduction='mean'
        )
        return loss

    def _compute_token_loss(
            self, next_token_logits: torch.Tensor, next_token_idxs: torch.Tensor, answer_mask: torch.Tensor
        ) -> torch.Tensor:
        batch_size, vocab_size = next_token_logits.shape
        assert next_token_idxs.shape == (batch_size,)

        if answer_mask is not None:
            assert answer_mask.shape == (vocab_size,)
            next_token_logits = torch.where(
                answer_mask[None],
                next_token_logits, torch.tensor(float('-inf')).to(device)
            )
                
        return torch.nn.functional.cross_entropy(
            input=next_token_logits,
            target=next_token_idxs,
            reduction='mean'
        )
    
    def __call__(self,
            input_ids: torch.Tensor,
            next_token_ids: torch.Tensor,
            logits: torch.Tensor,
            answer_mask: torch.Tensor,
        ) -> torch.Tensor:
        """Computes loss.

        Args:
            input_ids (int torch.Tensor): array of token IDs for inputs
            next_token_ids (int torch.Tensor): array of token IDs for the word
                that comes after the input
            logits (float torch.Tensor): logits for all output tokens, including
                the next one
            answer_mask (bool torch.Tensor): mask over tokens to remove irrelevant ones

        Returns: float torch.Tensor scalar, loss value (lower is better).
        """
        fluency_loss = (
            self._compute_fluency_loss(
                logits=logits,
                input_ids=input_ids
            )
        )

        print(f"[] loss for input_ids: {self.tokenizer.decode(input_ids[0])}")
        token_loss = (
            self._compute_token_loss(
                next_token_logits=logits[:, -1, :],
                next_token_idxs=next_token_ids,
                answer_mask=answer_mask,
            )
        )

        return token_loss + (self.gamma * fluency_loss)


class PrefixModel(nn.Module, abc.ABC):
    loss_func: PrefixLoss
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    
    def __init__(self, loss_func: PrefixLoss, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        self.loss_func = loss_func
        self.model = model
        self.tokenizer = tokenizer

    @functools.cached_property
    def id_to_word(self) -> Dict[int, str]:
        # track token-to-word mapping 
        return {num: word for word, num in self.tokenizer.vocab.items()}

    @property
    def transformer(self) -> nn.Module:
        return self.model._modules['transformer']

    @property
    def token_embedding(self) -> nn.Embedding:
        return self.transformer.wte
    
    @property
    def vocab_size(self) -> int:
        return self.token_embedding.weight.shape[0] # 50_257 for GPT2

    @property 
    def token_embedding_dim(self) -> int:
        return self.token_embedding.weight.shape[1] # often 768, or 2560 for some larger models

    def forward(self, input_ids: torch.Tensor, prefix_ids: Optional[torch.Tensor]) -> torch.Tensor:
        input_ids, embeddings = self.embed_input_ids(input_ids=input_ids, prefix_ids=prefix_ids)
        assert input_ids.shape == embeddings.shape[0:2]
        return input_ids, self.model(inputs_embeds=embeddings)
    
    def pre_epoch(self) -> None:
        return
    
    def post_epoch(self, dataloader: torch.utils.data.DataLoader, possible_answer_mask: torch.Tensor) -> None:
        return
    
    def compute_metrics(self) -> Dict[str, Any]:
        return {}

    @abc.abstractproperty
    def trainable_params(self) -> Iterable[nn.Parameter]:
        raise NotImplementedError()

    @abc.abstractmethod
    def embed_input_ids(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """To be implemented by subclasses -- embeds input ids and includes some sort of prefix,
        for example, in the case of prompt-tuning, by prepending a continuous embedding.
        """
        raise NotImplementedError()
    
    def init_continuous_prefix(self) -> nn.Parameter:
        # TODO: argparse for params
        N_TOKENS = 8 # TODO argparse for n_tokens
        return nn.Parameter(
            self.token_embedding.weight.mean(dim=0, keepdim=True)[None].repeat(1, N_TOKENS, 1), requires_grad=True
        )
        # return nn.Parameter(torch.randu((1, 1, emb_dim)), requires_grad=True).to(device)
        # return nn.Parameter(prefix_emb[:, 0, :], requires_grad=True).to(device)
    
    def init_discrete_prefix(self, num_tokens: int) -> nn.Parameter:
        # TODO: argparse for params
        # start_word_id = torch.tensor([self.tokenizer.vocab['the']], dtype=int)
        # start_word_id = torch.tensor([self.tokenizer.encode(' multiply')[0]], dtype=int)
        # start_word_id = torch.tensor([self.tokenizer.encode(' hello')[0]], dtype=int)
        # start_word_id = torch.tensor([self.tokenizer.encode('ogg')[0]], dtype=int)
        # start_word_id = torch.tensor([self.tokenizer.encode(' add')[0]], dtype=int)
        start_word_id = torch.tensor([self.tokenizer.encode('<|endoftext|>')[0]], dtype=int)
        print(f"start_word_id = {start_word_id}")
        return start_word_id.repeat((num_tokens,))
    
    def compute_loss(
            self,
            original_input_ids: torch.Tensor,
            next_token_ids: torch.Tensor,
            possible_answer_mask: Optional[torch.Tensor]
        ) -> Tuple[torch.Tensor, int]:
        """Computes loss using `self.loss_func`.
        
        Returns:
            loss (float torch.Tensor) -- the loss
            num_correct (int): number of examples where prediction was correct
        """
        input_ids, outputs = self.forward(input_ids=original_input_ids)

        next_token_logits = outputs.logits[:, -1, :]

        n_correct = (
            next_token_logits.argmax(dim=-1)
                ==
            next_token_ids
        ).int().sum()

        loss = self.loss_func(
            input_ids=input_ids,
            next_token_ids=next_token_ids,
            logits=outputs['logits'],
            answer_mask=possible_answer_mask
        )
        return loss, n_correct


class PromptTunedModel(PrefixModel):
    loss_func: PrefixLoss
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    prefix_embedding: nn.Parameter
    def __init__(self, loss_func: PrefixLoss, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__(loss_func=loss_func, model=model, tokenizer=tokenizer)
        self.prefix_embedding = self.init_continuous_prefix()

    def embed_input_ids(self, input_ids: torch.Tensor, prefix_ids: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert prefix_ids is None, "cannot provide custom prefix IDs for prompt-tuning"
        token_embeddings = self.token_embedding.forward(input_ids)
        return None, torch.cat(
            (self.prefix_embedding.repeat((len(input_ids), 1, 1)), token_embeddings), dim=1
        )

    @property
    def trainable_params(self) -> Iterable[nn.Parameter]:
        return [self.prefix_embedding]
    
    def compute_metrics(self) -> Dict[str, Any]:
        return {
            'embs': self.prefix_embedding.detach().cpu().numpy(),
            'grads': self.prefix_embedding.grad.detach().cpu().numpy(),
        }


VERBOSE = False # whether to print grads, etc.
TOP_K = 20 # for printing grads, etc.

class HotFlipPrefixModel(PrefixModel):
    loss_func: PrefixLoss
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    prefix_ids: torch.Tensor
    prefix_embedding: nn.Parameter
    def __init__(self, loss_func: PrefixLoss, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__(loss_func=loss_func, model=model, tokenizer=tokenizer)
        # HotFlip-specific parameters.
        self._min_loss = float('inf')
        self._num_tokens = 1 # TODO argparse for n_tokens
        self._num_candidates_per_prefix_token = 200 # TODO argparse for this too
        self._swap_token_idx = 0
        # 
        self.preprefix_ids = self.tokenizer.encode('The function to compute is', return_tensors='pt')
        self.prefix_ids = self.init_discrete_prefix(num_tokens=self._num_tokens)
        self.prefix_embedding = nn.Parameter(
            self.token_embedding.forward(self.prefix_ids), requires_grad=True
        )
    
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

    def _compute_loss(
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
        _input_ids, loss, n_correct = self._compute_loss(
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
        assert top_tokens_per_position.shape == (self._num_tokens, self._num_candidates_per_prefix_token)

        top_swap_tokens = top_tokens_per_position[token_idx, :]
        #
        # Get most likely tokens.
        #
        prefix_until_swap_ids = torch.cat(
            (self.preprefix_ids, self.prefix_ids[None, :token_idx]), dim=1
        ).to(device)
        swap_token_logits = self.model(self.preprefix_ids.to(device)).logits[:, -1, :]

        rvocab = {v: k for k,v in self.tokenizer.vocab.items()}
        # dist_sum = (swap_token_logits.log_softmax(dim=1) * .7 + (-1 * token_grads).log_softmax(dim=1))
        # for v in (swap_token_logits.log_softmax(dim=1) * .7 + (-1 * token_grads).log_softmax(dim=1)).topk(10).indices.flatten(): print(rvocab[v.item()])


        token_losses =(
            (swap_token_logits.log_softmax(dim=1) * 0.0 + (-1 * token_grads).log_softmax(dim=1))
        )
        top_swap_tokens = token_losses.argsort(descending=True).flatten()
        top_swap_tokens = top_swap_tokens[:self._num_candidates_per_prefix_token]
        # 
        # Evaluate candidates.
        # 
        all_candidate_losses = torch.zeros(self._num_candidates_per_prefix_token, dtype=float).to(device)
        best_loss = self._min_loss

        mask = torch.nn.functional.one_hot(
            torch.tensor(token_idx), num_classes=self._num_tokens
        ).bool().to(device)

        for batch in tqdm.tqdm(dataloader, desc='evaluating HotFlip candidates', colour='red', leave=False):
            # Loop in this order so we only tokenize each thing once.
            x_text = [f'. {prompt}' for prompt in batch['input']]
            y_text = [answer.replace('.', '').rstrip() for answer in batch['output']] # strip newlines and periods.
            #
            input_ids = self.tokenizer(x_text, return_tensors='pt')['input_ids'].to(device)
            next_token_ids = self.tokenizer(y_text, return_tensors='pt')['input_ids'].squeeze(dim=1).to(device)
            for candidate_idx in range(self._num_candidates_per_prefix_token):
                new_token_id = top_swap_tokens[candidate_idx]
                prefix_ids = torch.where(
                    mask, new_token_id, self.prefix_ids.to(device)
                ).to(device)
                with torch.no_grad():
                    _input_ids, loss, n_correct = self._compute_loss(
                        original_input_ids=input_ids,
                        next_token_ids=next_token_ids,
                        possible_answer_mask=possible_answer_mask,
                        prefix_ids=prefix_ids
                    )
                all_candidate_losses[candidate_idx] += loss

        #
        # Recreate prefix with min loss.
        #
        breakpoint()
        min_loss = all_candidate_losses[token_idx].min()
        best_candidate_idx = all_candidate_losses[token_idx].argmin()

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
        print(f'[Loss = {min_loss/len(dataloader):.2f}] Old prefix: {self.tokenizer.decode(self.prefix_ids)} // New prefix: {self.tokenizer.decode(best_prefix_ids)}')
        self._set_prefix_ids(best_prefix_ids)

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

        prefix_ids = prefix_ids.to(device).repeat((batch_size, 1))
        preprefix_ids = self.preprefix_ids.to(device).repeat((batch_size, 1))
        full_input_ids = torch.cat(
            (preprefix_ids, prefix_ids, input_ids), dim=1
        )
        # concatenate preprefix (fixed) + prefix (learned) + example
        outputs = torch.cat(
            (
                self.token_embedding.forward(preprefix_ids),
                self.prefix_embedding[None].repeat((batch_size, 1, 1)),
                self.token_embedding.forward(input_ids)
            ), dim=1
        )
        return full_input_ids, outputs


class GumbelPrefixModel(PrefixModel):
    loss_func: PrefixLoss
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    prefix_embedding: nn.Parameter

    def __init__(self, loss_func: PrefixLoss, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__(loss_func=loss_func, model=model, tokenizer=tokenizer)
        N_TOKENS = 3 # TODO argparse for n_tokens
        self.word_weights = nn.Parameter(
            torch.randn((1, N_TOKENS, self.vocab_size)), requires_grad=True
        )
        # TODO: argparse for tau
        # (lower tau -> more spiky)
        self.tau = 10
        # TODO: argparse for tau_anneal
        self.tau_anneal = 1.02

    @property
    def trainable_params(self) -> Iterable[nn.Parameter]:
        return [self.word_weights]
    
    def post_epoch(self, dataloader: torch.utils.data.DataLoader, possible_answer_mask: torch.Tensor) -> None:
        self.tau = self.tau / self.tau_anneal
        print(f"𝛕 = {self.tau:.2f}")

    def embed_input_ids(self, input_ids: torch.Tensor, prefix_ids: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert prefix_ids is None, "cannot provide custom prefix IDs for Gumbel"
        # word_weights_dist = (word_weights * 1000).softmax(dim=-1)
        prefix_embedding_words_dist = nn.functional.gumbel_softmax(
            self.word_weights.repeat((len(input_ids), 1, 1)), tau=self.tau, dim=-1, hard=False
        )
        
        print(
            "trying words:", self.tokenizer.decode(
                prefix_embedding_words_dist[0].argmax(dim=1).tolist()),
            "with prob", prefix_embedding_words_dist[0].max().item()
        )
        prefix_embedding = prefix_embedding_words_dist @ self.token_embedding.weight

        input_ids = torch.cat(
            (prefix_embedding_words_dist.argmax(dim=-1), input_ids), dim=1
        )
        outputs = torch.cat(
            # concatenate prefix + example
            (prefix_embedding, self.token_embedding.forward(input_ids)), dim=1
        )
        return input_ids, outputs
