from typing import Any, Dict, Iterable

import abc
import functools
import transformers
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PrefixTunedModel(torch.nn.Module, abc.ABC):
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    
    def __init__(self, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    @functools.cached_property
    def id_to_word(self) -> Dict[int, str]:
        # track token-to-word mapping 
        return {num: word for word, num in self.tokenizer.vocab.items()}

    @property
    def transformer(self) -> torch.nn.Module:
        return self.model._modules['transformer']

    @property
    def token_embedding(self) -> torch.nn.Embedding:
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
        embeddings = self.embed_input_ids(input_ids=input_ids)
        return self.model(inputs_embeds=embeddings)
    
    def pre_epoch(self) -> None:
        return
    
    def post_epoch(self) -> None:
        return
    
    def compute_metrics(self) -> Dict[str, Any]:
        return {}

    @abc.abstractproperty
    def trainable_params(self) -> Iterable[torch.nn.Parameter]:
        raise NotImplementedError()

    @abc.abstractmethod
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """To be implemented by subclasses -- embeds input ids and includes some sort of prefix,
        for example, in the case of prompt-tuning, by prepending a continuous embedding.
        """
        raise NotImplementedError()
    
    def init_continuous_prefix_embedding(self) -> torch.nn.Parameter:
        # TODO: argparse for params
        N_TOKENS = 1 # TODO argparse for n_tokens
        return torch.nn.Parameter(
            self.token_embedding.weight.mean(dim=0, keepdim=True)[None].repeat(1, N_TOKENS, 1), requires_grad=True
        )
        # return torch.nn.Parameter(torch.randu((1, 1, emb_dim)), requires_grad=True).to(device)
        # return torch.nn.Parameter(prefix_emb[:, 0, :], requires_grad=True).to(device)
    
    def init_discrete_prefix_embedding(self) -> torch.nn.Parameter:
        # TODO: argparse for params
        N_TOKENS = 1 # TODO argparse for n_tokens
        start_word_id = torch.tensor(self.tokenizer.vocab['the'], dtype=int)
        return torch.nn.Parameter(
            torch.nn.functional.one_hot(
                start_word_id, num_classes=self.vocab_size
            )
            .reshape((1, 1, self.vocab_size))
            .repeat((1, N_TOKENS, 1))
            .float(),
            requires_grad=True
        )


class PromptTunedModel(PrefixTunedModel):
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    trainable_prefix_embedding: torch.nn.Parameter
    def __init__(self, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__(model=model, tokenizer=tokenizer)
        self.trainable_prefix_embedding = self.init_continuous_prefix_embedding()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embedding.forward(input_ids)
        return torch.cat(
            (self.trainable_prefix_embedding.repeat((len(input_ids), 1, 1)), token_embeddings), dim=1
        )

    @property
    def trainable_params(self) -> Iterable[torch.nn.Parameter]:
        return [self.trainable_prefix_embedding]
    
    def compute_metrics(self) -> Dict[str, Any]:
        return {
            'embs': self.trainable_prefix_embedding.detach().cpu().numpy(),
            'grads': self.trainable_prefix_embedding.grad.detach().cpu().numpy(),
        }


VERBOSE = False
TOP_K = 20 # for printing grads, etc.

class HotFlipPrefixTunedModel(PrefixTunedModel):
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    trainable_prefix_embedding: torch.nn.Parameter
    def __init__(self, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__(model=model, tokenizer=tokenizer)
        self.trainable_prefix_embedding = self.init_discrete_prefix_embedding()

    def pre_epoch(self) -> None:
        # Print closest tokens at the beginning of each epoch.
        if VERBOSE:
            print("*" *  30)
            print(f"Epoch {epoch}. Closest tokens to '{prefix_str}':")
            word_distances =  ((wte.weight - self.trainable_prefix_embedding.reshape(1, emb_dim))**2).sum(1)
            assert word_distances.shape == (50_257,)
            topk_closest_words = distances = word_distances.topk(k=TOP_K, largest=False)
            for _id, _dist in zip(topk_closest_words.indices.cpu().tolist(), topk_closest_words.values.cpu().tolist()):
                print(f'\t{self.id_to_word[_id]} ({_id}): {_dist:.3f}')
            print("*" * 30)

    def post_epoch(self) -> None:
        if VERBOSE:
            # Print gradient information.
            token_grads = wte.weight.mv(self.trainable_prefix_embedding.grad.flatten())
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

    @property
    def trainable_params(self) -> Iterable[torch.nn.Parameter]:
        return [self.trainable_prefix_embedding]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError() # TODO


class GumbelPrefixTunedModel(PrefixTunedModel):
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    trainable_prefix_embedding: torch.nn.Parameter

    def __init__(self, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__(model=model, tokenizer=tokenizer)
        N_TOKENS = 64 # TODO argparse for n_tokens
        self.word_weights = torch.nn.Parameter(
            torch.randn((1, N_TOKENS, self.vocab_size)), requires_grad=True
        )
        # TODO: argparse for tau
        # low tau -> very spiky
        self.tau = 10
        # TODO: argparse for tau_anneal
        self.tau_anneal = 1.03

    @property
    def trainable_params(self) -> Iterable[torch.nn.Parameter]:
        return [self.word_weights]
    
    def post_epoch(self) -> None:
        self.tau = self.tau / self.tau_anneal
        print(f"𝛕 = {self.tau:.2f}")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # word_weights_dist = (word_weights * 1000).softmax(dim=-1)
        prefix_embedding_words_dist = torch.nn.functional.gumbel_softmax(
            self.word_weights.repeat((len(input_ids), 1, 1)), tau=self.tau, dim=-1, hard=False
        )
        
        print(
            "trying words:", self.tokenizer.decode(
                prefix_embedding_words_dist[0].argmax(dim=1).tolist()),
            "with prob", prefix_embedding_words_dist[0].max().item()
        )
        prefix_embedding = prefix_embedding_words_dist @ self.token_embedding.weight

        # concatenate prefix + example
        return torch.cat(
            (
                prefix_embedding,
                self.token_embedding.forward(input_ids)
            ), 
            dim=1
        )
