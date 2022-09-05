from typing import Dict, Iterable

import abc
import functools
import transformers
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PrefixTunedModel(torch.nn.Module, abc.ABC):
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    
    def __init__(self, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
        super(torch.nn.Module).__init__()
        self.model = model
        self.tokenizer = tokenizer

    @functools.cached_property
    def id_to_word(self) -> Dict[int, str]:
        # track token-to-word mapping 
        return {num: word for word, num in self.tokenizer.vocab.items()}
    
    @property
    def vocab_size(self) -> int:
        return self.token_embedding_matrix.shape[0] # 50_257 for GPT2

    @property 
    def token_embedding_dim(self) -> int:
        return self.token_embedding_matrix.shape[1] # often 768, or 2560 for some larger models

    @property
    def transformer(self) -> torch.nn.Module:
        return self.model._modules['transformer']

    @property
    def token_embedding_matrix(self) -> torch.nn.Embedding:
        return self.trasnformer.wte
    
    def forward_text(self, text: Iterable[str]) -> torch.Tensor:
        tokenized_text = self.tokenizer(text, return_tensors='pt')['input_ids'].to(device)
        return self.forward(**tokenized_text)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embed_input_ids(input_ids=input_ids)
        return self.transformer(inputs_embeds=emb)
    
    def pre_epoch(self) -> None:
        return
    
    def post_epoch(self) -> None:
        return

    @abc.abstractmethod
    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """To be implemented by subclasses -- embeds input ids and adds some sort of prefix,
        for example a trainable continuous embedding in the case of prompt-tuning.
        """
        raise NotImplementedError()
    
    def init_continuous_prefix_embedding(self) -> torch.nn.Parameter:
        # TODO: argparse for params
        N_TOKENS = 1 # TODO argparse for n_tokens
        return torch.nn.Parameter(
            wte.weight.mean(dim=0, keepdim=True)[None].repeat(1, N_TOKENS, 1), requires_grad=True
        )
        # return torch.nn.Parameter(torch.randu((1, 1, emb_dim)), requires_grad=True).to(device)
        # return torch.nn.Parameter(prefix_emb[:, 0, :], requires_grad=True).to(device)
    
    def init_discrete_prefix_embedding(self) -> torch.nn.Parameter:
        # TODO: argparse for params
        N_TOKENS = 1 # TODO argparse for n_tokens
        return torch.nn.Parameter(
            torch.nn.functional.one_hot(
                    torch.tensor(tokenizer.vocab['the'], dtype=int), num_classes=vocab_size
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

    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embedding.forward(
            input_ids=input_ids
        )
        return torch.cat(
            (self.trainable_prefix_embedding.repeat((len(y_text), 1, 1)), token_embeddings), dim=1
        )


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
            word_distances =  ((wte.weight - trainable_prefix_emb.reshape(1, emb_dim))**2).sum(1)
            assert word_distances.shape == (50_257,)
            topk_closest_words = distances = word_distances.topk(k=TOP_K, largest=False)
            for _id, _dist in zip(topk_closest_words.indices.cpu().tolist(), topk_closest_words.values.cpu().tolist()):
                print(f'\t{self.id_to_word[_id]} ({_id}): {_dist:.3f}')
            print("*" * 30)

    def post_epoch(self) -> None:
        if VERBOSE:
            # Print gradient information.
            token_grads = wte.weight.mv(trainable_prefix_emb.grad.flatten())
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

    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError() # TODO


class GumbelPrefixTunedModel(PrefixTunedModel):
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    trainable_prefix_embedding: torch.nn.Parameter

    def __init__(self, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__(model=model, tokenizer=tokenizer)
        N_TOKENS = 1 # TODO argparse for n_tokens
        self.trainable_prefix_embedding = self.init_discrete_prefix_embedding()

    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        ex_inputs = tokenizer(full_text, return_tensors='pt').to(device)
        # print(ex_inputs)
        ex_embs = wte.forward(ex_inputs['input_ids'].to(
            device)).to(device)

        # word_weights_dist = (word_weights * 1000).softmax(dim=-1)
        # # low tau -> very spiky
        word_weights_dist = torch.nn.functional.gumbel_softmax(
            word_weights, tau=1_000, dim=-1, hard=False
        )
        word_weights_dist = word_weights
        # print("words:", word_weights_dist.argmax(-1).tolist())
        # breakpoint()
        # print("trying words:", word_weights_dist[0].argmax(dim=1).tolist(), "with prob", word_weights_dist[0].max().item())
        word_emb = word_weights_dist[:len(y_text)] @ wte.weight
        # word_emb = word_weights[:len(y_text)].mean(dim=0, keepdim=True).softmax(dim=1) @ wte.weight

        # concatenate prefix + example
        return torch.cat(
            (
                # trainable_prefix_emb.repeat((len(y_text), 1, 1)),
                word_emb,
                # word_emb[None].repeat((len(y_text), 1, 1)), # add fake sequence dim
                # word_emb.unsqueeze(dim=1), # add fake sequence dim
                ex_embs
            ), 
            dim=1
        )
