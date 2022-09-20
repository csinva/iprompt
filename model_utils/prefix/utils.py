from typing import Any, Dict, List, Iterable, Optional, Tuple

import abc
import argparse
import dataclasses
import functools
import random
import transformers
import torch
from torch.utils.data import DataLoader
from torch import nn
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEBUG_LOSS = False


def get_token_replacements_single_mask(
    dataloader: DataLoader, model: transformers.AutoModelForMaskedLM,
    tokenizer: transformers.AutoTokenizer, init_prefix_template: str, num_candidates: int)-> List[str]:
    """Given a template like `{mask} the numbers`, returns the `num_candidates` most likely
    single-token replacements for `{mask}` given `model`.
    """
    single_mask_prefix_str = init_prefix_template.format(mask=tokenizer.mask_token)
    all_mask_probs = torch.zeros((tokenizer.vocab_size,), dtype=float).to(device)
    for idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        full_text = [f'{single_mask_prefix_str} {input_text}' for input_text in batch['text']]
        if idx == 0:
            print('Sample input: ', full_text[0])
        inputs = tokenizer(full_text, return_tensors='pt', padding='longest')
        with torch.no_grad():
            outputs = model(**inputs.to(device))
        mask_idxs = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero()
        # TODO: how to do this better in torch?
        mask_probs = outputs.logits[mask_idxs[:, 0], mask_idxs[:, 1]].log_softmax(dim=1)
        all_mask_probs += mask_probs.sum(dim=0)
        
    prefix_idxs = all_mask_probs.topk(num_candidates).indices
    return [init_prefix_template.format(mask=tokenizer.decode(idx)) for idx in prefix_idxs]


def get_prefix_from_mlm(
        dataloader: DataLoader,
        mlm_name: str,
        num_candidates: int,
        template: str
    ) -> List[str]:
    """ Getting prefix from MLM."""
    mlm = transformers.RobertaForMaskedLM.from_pretrained(mlm_name).to(device)
    mlm_tokenizer = transformers.AutoTokenizer.from_pretrained(mlm_name)
    # template = "{mask} the two numbers to get the answer."
    # template = "{mask} the input number to get the answer."
    # template = "Return the{mask} of the input."

    candidates = get_token_replacements_single_mask(
        dataloader=dataloader,
        model=mlm, tokenizer=mlm_tokenizer,
        init_prefix_template=template,
        num_candidates=num_candidates
    )
    mlm.to('cpu') # no need for mlm on GPU anymore
    return candidates


def compute_log_ppl_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Computes LM perplexity loss given logits for next tokens and original input IDs.
    Exponentiate this quantity if you want the actual perplexity.
    """
    # logits gives us the probability of each token that comes after each token in input_ids.
    # so they have the same shape. But we only want to compute ppl using the tokens we have,
    # i.e. not the first true token (which we don't have logits for) or the last predicted token
    # (which we don't know the true id for). so we have to shift each by one index.
    assert logits.shape[0:2] == input_ids.shape
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
        return compute_log_ppl_loss(logits=logits, input_ids=input_ids)

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
    
    def __call__(
            self,
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

        loss = token_loss + (self.gamma * fluency_loss)
        if DEBUG_LOSS: 
            print(f">> loss for input string: {self.tokenizer.decode(input_ids[0])}")
            print(f"\tLoss = {loss:.3f}")
        return loss


class PrefixModel(nn.Module, abc.ABC):
    args: argparse.Namespace
    loss_func: PrefixLoss
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    
    def __init__(self, args: argparse.Namespace, loss_func: PrefixLoss, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, preprefix: str):
        super().__init__()
        self.args = args
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
    
    def prepare_batch(self, batch: Dict[str, str]) -> Tuple[str, str]:
        """Preprocesses text from `batch['input']` and `batch['output']` for inputting into prefix model.
        """
        x_text = [f'. {prompt}' for prompt in batch['input']]
        y_text = [answer.replace('.', '').rstrip() for answer in batch['output']] # strip newlines and periods.
        return x_text, y_text

    def forward(
            self,
            input_ids: torch.Tensor,
            prefix_ids: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        new_input_ids, embeddings = self.embed_input_ids(
            input_ids=input_ids, prefix_ids=prefix_ids
        )
        assert new_input_ids.shape == embeddings.shape[0:2]
        return new_input_ids, self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
    
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
    
    def init_continuous_prefix(self, num_tokens: int) -> nn.Parameter:
        return nn.Parameter(
            self.token_embedding.weight.mean(dim=0, keepdim=True)[None].repeat(1, num_tokens, 1), requires_grad=True
        )
        # return nn.Parameter(torch.randu((1, 1, emb_dim)), requires_grad=True).to(device)
        # return nn.Parameter(prefix_emb[:, 0, :], requires_grad=True).to(device)
    
    def init_discrete_prefix(self, num_tokens: int) -> nn.Parameter:
        # TODO: argparse for starting token
        # start_word_id = torch.tensor([self.tokenizer.encode(' multiply')[0]], dtype=int)
        # start_word_id = torch.tensor([self.tokenizer.encode(' hello')[0]], dtype=int)
        # start_word_id = torch.tensor([self.tokenizer.encode('ogg')[0]], dtype=int)
        # start_word_id = torch.tensor([self.tokenizer.encode(' add')[0]], dtype=int)
        start_word_id = torch.tensor([self.tokenizer.encode('<|endoftext|>')[0]], dtype=int)
        # start_word_id = torch.tensor([self.tokenizer.vocab['the']], dtype=int)
        print(f"start_word_id = {start_word_id}")
        return start_word_id.repeat((num_tokens,))
    
    def compute_loss_and_call_backward(
            self,
            x_tokenized: transformers.BatchEncoding,
            y_tokenized: transformers.BatchEncoding,
            possible_answer_mask: Optional[torch.Tensor],
            full_text_tokenized: Optional[transformers.BatchEncoding] = None
        ) -> Tuple[torch.Tensor, int]:
        """Computes loss using `self.loss_func`.
        
        Returns:
            loss (float torch.Tensor) -- the loss
            num_correct (int): number of examples where prediction was correct
        """
        original_input_ids = x_tokenized.input_ids
        next_token_ids = y_tokenized.input_ids[:, 0] # only compute loss over next token

        input_ids, outputs = self.forward(input_ids=original_input_ids, prefix_ids=None)

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
        loss.backward()
        return loss, n_correct