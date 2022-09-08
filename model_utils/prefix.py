from typing import Any, Dict, Iterable, Tuple

import abc
import dataclasses
import functools
import transformers
import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclasses.dataclass
class PrefixLoss:
    """Computes next-token-prediction loss with optional language modeling component.
    """
    gamma: float

    def _compute_fluency_loss(
            self, logits: torch.Tensor, input_ids: torch.Tensor
        ) -> torch.Tensor:
        if self.gamma == 0:
            return torch.tensor(0.0).to(device)
        log_probs = logits.log_softmax(dim=-1)
        num_input_words = input_ids.shape[1]
        log_probs_for_input = log_probs[:, -1-num_input_words:-1, :]
        input_log_probs = torch.gather(
            log_probs_for_input, dim=2, index=input_ids[...,None].to(device)
        )
        return -1 * input_log_probs.mean()

    def _compute_token_loss(
            self, next_token_logits: torch.Tensor, next_token_idxs: torch.Tensor, answer_mask: torch.Tensor
        ) -> torch.Tensor:
        batch_size, vocab_size = next_token_logits.shape
        assert next_token_idxs.shape == (batch_size,)
        assert answer_mask.shape == (vocab_size,)
        next_token_logits = torch.where(
            answer_mask[None], next_token_logits, torch.tensor(float('-inf')).to(device)
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
    
    def forward_text(self, text: Iterable[str]) -> torch.Tensor:
        tokenized_text = self.tokenizer(text, return_tensors='pt')
        return self.forward(
            input_ids=tokenized_text['input_ids'].to(device)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids, embeddings = self.embed_input_ids(input_ids=input_ids)
        return input_ids, self.model(inputs_embeds=embeddings)
    
    def pre_epoch(self) -> None:
        return
    
    def post_epoch(self) -> None:
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
    
    def init_discrete_prefix(self) -> nn.Parameter:
        # TODO: argparse for params
        N_TOKENS = 64 # TODO argparse for n_tokens
        start_word_id = torch.tensor([self.tokenizer.vocab['the']], dtype=int)
        return start_word_id.repeat((N_TOKENS,))
    
    def compute_loss(self, text: str, next_token_ids: torch.Tensor, possible_answer_mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Computes loss using `self.loss_func`.
        
        Returns:
            loss (float torch.Tensor) -- the loss
            num_correct (int): number of examples where prediction was correct
        """
        input_ids, outputs = self.forward_text(text=text)
        n_correct = (
            outputs['logits'][:, -1, :].argmax(dim=-1)
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

    def embed_input_ids(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


VERBOSE = False
TOP_K = 20 # for printing grads, etc.

class HotFlipPrefixModel(PrefixModel):
    loss_func: PrefixLoss
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    prefix_ids: torch.Tensor
    prefix_embedding: nn.Parameter
    def __init__(self, loss_func: PrefixLoss, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__(loss_func=loss_func, model=model, tokenizer=tokenizer)
        self.beam_width = 1 # TODO argparse for beam_width
        self.num_candidates = 10
        self.prefix_ids = self.init_discrete_prefix()
        self.prefix_embedding = nn.Parameter(
            self.token_embedding.forward(self.prefix_ids), requires_grad=True
        )
    
    def _set_prefix_ids(self, new_ids: torch.Tensor) -> None:
        self.prefix_ids = new_ids
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

    def post_epoch(self) -> None:
        # variables: V (vocab), d (embedding dim), n (num prefix tokens)
        token_grads = torch.einsum('nd,vd->nv', self.prefix_embedding.grad, self.token_embedding.weight)
        if VERBOSE:
            # Print gradient information.
            print(f"Epoch {epoch}. Most negative grads for x:")
            assert token_grads.shape == (50_257, )
            topk_smallest_grads = distances = token_grads.topk(k=TOP_K, largest=False)
            for _id, _dist in zip(topk_smallest_grads.indices.cpu().tolist(), topk_smallest_grads.values.cpu().tolist()):
                print(f'\t{self.id_to_word[_id]} ({_id}): {_dist:.3f}')
            print("*" * 30)
            print(f"Epoch {epoch}. Most positive grads for x:")
            topk_largest_grads = distances = token_grads.topk(k=TOP_K, largest=True)
            for _id, _dist in zip(topk_largest_grads.indices.cpu().tolist()[::-1], topk_largest_grads.values.cpu().tolist()[::-1]):
                print(f'\t{self.id_to_word[_id]} ({_id}): {_dist:.3f}')
            print("*" * 30)
        ############################################################
        # TODO: Argparse for different search strategies.
        # Flip word with most negative grad.
        # token_to_flip_idx = token_grads.min(dim=1).values.argmin()
        # new_token_id = token_grads[token_to_flip_idx].argmin()
        # new_tokens = self.prefix_ids.cpu().numpy()
        # new_tokens[token_to_flip_idx] = new_token_id
        # self._set_prefix_ids(torch.tensor(new_tokens).to(device))
        ############################################################
        # Flip all words to token with most negative grad.
        self._set_prefix_ids(token_grads.argmin(dim=1))
        ############################################################
        # TODO >> FLIP here
        print("new words:", self.tokenizer.decode(self.prefix_ids.tolist()))
    
    @property
    def prefix_embedding_token_ids(self) -> torch.Tensor:
        return self.prefix_embedding.argmax(dim=-1)

    @property
    def trainable_params(self) -> Iterable[nn.Parameter]:
        return [self.prefix_embedding]

    def embed_input_ids(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(input_ids)
        input_ids = torch.cat(
            (self.prefix_embedding_token_ids.repeat((batch_size, 1)), input_ids), dim=1
        )
        outputs = torch.cat(
            # concatenate prefix + example
            (self.prefix_embedding[None].repeat((batch_size, 1, 1)), self.token_embedding.forward(input_ids)), dim=1
        )
        return input_ids, outputs


class GumbelPrefixModel(PrefixModel):
    loss_func: PrefixLoss
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    prefix_embedding: nn.Parameter

    def __init__(self, loss_func: PrefixLoss, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__(loss_func=loss_func, model=model, tokenizer=tokenizer)
        N_TOKENS = 8 # TODO argparse for n_tokens
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
    
    def post_epoch(self) -> None:
        self.tau = self.tau / self.tau_anneal
        print(f"𝛕 = {self.tau:.2f}")

    def embed_input_ids(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
