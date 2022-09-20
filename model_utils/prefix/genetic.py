from typing import Dict, List, Optional, Tuple, Union

import argparse
import collections
import heapq
import random
import torch
import transformers

from .autoprompt import AutoPrompt
from .utils import device, PrefixLoss, PrefixModel


"""
docs for generate():
    https://huggingface.co/docs/transformers/v4.22.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate


generation example:
    tokenizer.decode(lm.generate(input_ids=None, max_length=15, temperature=0.8, top_p=0.8, do_sample=True)[0])
"""

def mean(_list: List[Union[int, float]]) -> float:
    return sum(_list) / len(_list)

class GeneticAutoPrompt(AutoPrompt):
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
            args=args, loss_func=loss_func, model=model, tokenizer=tokenizer, preprefix=''
        )
        self.preprefix_ids = torch.tensor([], dtype=int).to(device)
        self.tokenizer.add_special_tokens = False
        # GeneticAutoPrompt-specific parameters
        # tuple (input_ids) -> float (loss)
        self._genetic_pool_loss = {}
        self._genetic_pool_all_losses = collections.defaultdict(list)
        # tuple (input_ids) -> int (n_correct)
        self._genetic_pool_accuracy = {}
        self._genetic_pool_all_accuracy = collections.defaultdict(list)
        ####################################################################
        # TODO argparse for GA-specific hparams
        self._top_k_pop_sample = 32 # sample next population from this num of top things
        self._pop_size = 8
        self._num_mutations_per_ex = 4 # num mutations for each population item
        self._num_random_generations = 4 # extra random examples to throw in there
        self._generation_temp = 1.0
        self._generation_top_p = 1.0
        self._generation_repetition_penalty = 20.0 # 1 means no penalty
        self._pop_initialized = False
        self._pop_selection_criterion = 'combined' # in ['loss', 'acc', 'combined']
        self._generation_bad_words_ids = [
            self.tokenizer.encode('\n'),
            self.tokenizer.encode('\n\n'),
            self.tokenizer.encode('\n\n\n')
        ]
        ####################################################################
        self._pre_data_token_ids = self.tokenizer("Data:\n\n", return_tensors='pt').input_ids.to(device)
        self._post_data_token_ids = self.tokenizer("Prompt:", return_tensors='pt').input_ids.to(device)
        ####################################################################
        self._verbose = True
    
    def _initialize_pop_once(self, full_text_ids: torch.Tensor):
        if self._pop_initialized: return

        while len(self._genetic_pool_loss) < self._pop_size:
            conditional_input_ids = random.choice(full_text_ids)[None]
            num_conditional_tokens = conditional_input_ids.numel()
            input_ids = self._generate(
                input_ids=conditional_input_ids,
                num_conditional_tokens=num_conditional_tokens
            )
            input_ids = input_ids[0, num_conditional_tokens:]
            input_ids = tuple(input_ids.cpu().tolist())
            assert len(input_ids) == self._num_tokens
            self._genetic_pool_loss[input_ids] = 10_000.0
            self._genetic_pool_accuracy[input_ids] = 0
        print('Original population:', self._genetic_pool_loss.keys())

        self._pop_initialized = True
    
    def _generate(self, input_ids: torch.Tensor, num_conditional_tokens: int) -> torch.Tensor:
        """Generates some text using the model and preset hparams.

        If `num_conditional_tokens` > 0, generates extra text because there was an additional
        prefix set.
        """
        output_length = self._num_tokens + num_conditional_tokens
        attention_mask = ~(input_ids == self.tokenizer.pad_token_id)
        assert attention_mask.shape == input_ids.shape
        
        g = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            min_length=output_length,
            max_length=output_length,
            temperature=self._generation_temp,
            top_p=self._generation_top_p,
            repetition_penalty=self._generation_repetition_penalty,
            bad_words_ids=self._generation_bad_words_ids,
            do_sample=True
        )

        if self._verbose:
            # Print a random one (but remove padded tokens and newlines)
            idx = random.choice(range(len(input_ids)))
            idx_attention_mask = torch.cat(
                (attention_mask[idx], torch.ones(self._num_tokens).to(device)), dim=0
            ).bool()
            random_sentence_ids = g[idx]
            print(">", self.tokenizer.decode(random_sentence_ids).replace('\n', '\\n'))

        return g
    
    def _select_pop_top_k(self, k: int, min_ocurrences: int = None) -> List[Tuple[int]]:
        prefixes = self._genetic_pool_all_accuracy.keys()
        if min_ocurrences:
            prefixes = {
                prefix for prefix in prefixes
                if len(self._genetic_pool_all_accuracy[prefix]) > min_ocurrences
            }

        criterion = self._pop_selection_criterion
        if criterion == 'loss':
            # sort by min loss
            population = [(loss, prefix_ids) for prefix_ids, loss in items()]
            topk_pop = heapq.nsmallest(k, population)
        elif criterion == 'combined':
            population = [
                ((self._genetic_pool_accuracy[prefix_ids], (-1 * loss)), prefix_ids)
                for prefix_ids, loss in self._genetic_pool_accuracy.items()
            ]
            topk_pop = heapq.nlargest(k, population)
        else:
            # sort by max acc
            population = [(loss, prefix_ids) for prefix_ids, loss in self._genetic_pool_accuracy.items()]
            topk_pop = heapq.nlargest(k, population)
            
        return [prefix_ids for _, prefix_ids in topk_pop]
    
    def _print_pop(self, top_k: int = 6) -> None:
        print((" " * 40), ("*" * 20), "Population", ("*" * 20))
        for token_ids in self._select_pop_top_k(k=top_k, min_ocurrences=3):
            prefix_str = "{:>70}".format(self.tokenizer.decode(list(token_ids)).replace("\n", "\\\\n"))
            loss_str = f"{self._genetic_pool_loss[token_ids]:.3f}"
            acc_str = f"{self._genetic_pool_accuracy[token_ids]:.2f}"
            print(prefix_str, "\t\t", loss_str, "\t\t", acc_str)
        print()
    
    def _get_population(self) -> torch.Tensor:
        # TODO sample from population instead of taking top-k?
        # population = self._select_pop_top_k(k=self._pop_size)
        population_pool = self._select_pop_top_k(k=self._top_k_pop_sample)
        population = random.sample(population_pool, self._pop_size)
        return torch.tensor(population).to(device)
    
    def _get_population_and_random_generations(self) -> torch.Tensor:
        raise NotImplementedError('doesn\'t work yet with conditional (TODO!)')
        population = self._get_population()
        starting_tensor = (
            torch.tensor([self.tokenizer.bos_token_id], dtype=int).to(device).repeat((self._num_random_generations, 1))
        )
        random_population = self._generate(starting_tensor)
        full_population = torch.cat((population, random_population), dim=0)
        assert full_population.shape == (
            self._pop_size + self._num_random_generations,
            self._num_tokens+1
        )
        return full_population
    
    def _mutate(self, population_input_ids: torch.Tensor, full_text_ids: torch.Tensor) -> List[torch.Tensor]:
        """Mutates a population of prefixes.

        Truncates to a random place and then generates new options
        to try.

        Args:
            population_input_ids (int torch.Tensor): input IDs for each prefix in population
            full_text_ids (int torch.Tensor): input IDs for each data item in the batch. Intended
                be used to do prefix generation conditioned on data
        """
        input_ids = population_input_ids.repeat((self._num_mutations_per_ex, 1))
        truncate_position = random.randint(0, self._num_tokens-1)

        rand_idxs = torch.randint(low=0, high=len(full_text_ids), size=(len(input_ids), ))
        random_full_text_ids = full_text_ids[rand_idxs]
        conditional_input_ids = torch.cat((random_full_text_ids, input_ids[:, :truncate_position]), dim=1)

        num_conditional_tokens = full_text_ids.shape[1]
        new_input_ids = self._generate(
            input_ids=conditional_input_ids,
            num_conditional_tokens=num_conditional_tokens
        )
        if (new_input_ids==self.tokenizer.pad_token_id).any():
            breakpoint()
        # Split off the conditional part, we only want the prefix part, which
        # starts after the conditional part.
        new_input_ids = new_input_ids[:, num_conditional_tokens:]
        
        # TODO consider adding crossover (combining spans?) here.

        return new_input_ids
    
    def _score_population(
            self, 
            x_tokenized: transformers.BatchEncoding,
            y_tokenized: transformers.BatchEncoding,
            population_input_ids: torch.Tensor,
            possible_answer_mask: torch.Tensor
        ):
        """Scores a population of prefixes and updates `self._genetic_pool`."""
        pop_size = len(population_input_ids)
        all_candidate_losses = torch.zeros(pop_size, dtype=float).to(device)
        all_accuracy = torch.zeros(pop_size, dtype=float).to(device)
        for i in range(pop_size):
            with torch.no_grad():
                prefix_ids = torch.cat(
                    (self._post_data_token_ids.flatten(), population_input_ids[i]), dim=0
                )
                _cand_input_ids, cand_loss, cand_n_correct = (
                    self._compute_loss_with_set_prefix(
                        original_input_ids=x_tokenized.input_ids,
                        next_token_ids=y_tokenized.input_ids[:, 0],
                        possible_answer_mask=possible_answer_mask,
                        prefix_ids=prefix_ids,
                    )
                )
                cand_accuracy = cand_n_correct / len(x_tokenized.input_ids)
            all_candidate_losses[i] += cand_loss
            all_accuracy[i] += cand_accuracy
        
        for i in range(pop_size):
            new_pop_input_ids = tuple(population_input_ids[i].cpu().tolist())
            assert len(new_pop_input_ids) == self._num_tokens

            # todo abstract these data strcutures into a class
            self._genetic_pool_all_losses[new_pop_input_ids].append(all_candidate_losses[i].item())
            self._genetic_pool_loss[new_pop_input_ids] = mean(self._genetic_pool_all_losses[new_pop_input_ids])
            self._genetic_pool_all_accuracy[new_pop_input_ids].append(all_accuracy[i].item())
            self._genetic_pool_accuracy[new_pop_input_ids] = mean(self._genetic_pool_all_accuracy[new_pop_input_ids])
    
    def _create_full_text_ids(
        self, full_text_input_ids: torch.Tensor) -> torch.Tensor:
        """Creates input for generating explanation.

        Takes tokenized inputs (like: "Input: 7 8 Output: 15")
        and makes a full string that looks like "Data:\n\n Input: .... 15 \n\nExplanation:\n\n",
        using whatever template is defined by pre-data and post-data.
        """
        B = len(full_text_input_ids)
        pre_data = self._pre_data_token_ids.repeat((B, 1)).to(device)
        post_data = self._post_data_token_ids.repeat((B, 1)).to(device)
        output = torch.cat((pre_data, full_text_input_ids, post_data), dim=1)
        return output

    def compute_loss_and_call_backward(
            self,
            x_tokenized: transformers.BatchEncoding,
            y_tokenized: transformers.BatchEncoding,
            possible_answer_mask: torch.Tensor,
            full_text_tokenized: Optional[transformers.BatchEncoding] = None
        ) -> Tuple[torch.Tensor, int]:
        """Returns the loss from the best example in the population

        Note: does not call loss.backward()
        
        Returns:
            loss (float torch.Tensor) -- the loss
            num_correct (int): number of examples where prediction was correct
        """
        full_text_ids = self._create_full_text_ids(
            full_text_input_ids=full_text_tokenized.input_ids,
        )
        self._initialize_pop_once(full_text_ids=full_text_ids)
        self._print_pop()

        # Grab new population
        population_input_ids = self._get_population()
        # TODO: Consider restoring random generations from below
        # population_input_ids = self._get_population_and_random_generations()
        mutated_population_input_ids = self._mutate(
            population_input_ids=population_input_ids, full_text_ids=full_text_ids
        )
        population_input_ids = torch.cat(
            (population_input_ids, mutated_population_input_ids), dim=0
        )
        # Re-score new guys
        self._score_population(
            x_tokenized=x_tokenized,
            y_tokenized=y_tokenized,
            population_input_ids=population_input_ids,
            possible_answer_mask=possible_answer_mask
        )
        # Return best one
        best_prefix_ids = min(self._genetic_pool_loss, key=self._genetic_pool_loss.get)
        best_prefix_last_loss = self._genetic_pool_all_losses[best_prefix_ids][-1]
        best_prefix_last_accuracy = self._genetic_pool_all_accuracy[best_prefix_ids][-1]
        best_prefix_last_n_correct = best_prefix_last_accuracy * len(full_text_tokenized.input_ids)

        return best_prefix_last_loss, best_prefix_last_n_correct
        
    def post_epoch(self, dataloader: torch.utils.data.DataLoader, possible_answer_mask: torch.Tensor) -> None:
        # 
        # Get candidate IDs for every position.
        # 
        pass