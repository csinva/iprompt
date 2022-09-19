from typing import List, Tuple

import argparse
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
            args=args, loss_func=loss_func, model=model, tokenizer=tokenizer, preprefix=preprefix
        )
        # GeneticAutoPrompt-specific parameters
        # tuple (input_ids) -> float (loss)
        self._genetic_pool_loss = {}
        # tuple (input_ids) -> int (n_correct)
        self._genetic_pool_n_correct = {}
        ####################################################################
        # TODO argparse for GA-specific hparams
        self._top_k_pop_sample = 32 # sample next population from this num of top things
        self._pop_size = 15
        self._num_mutations_per_ex = 4 # num mutations for each population item
        self._num_random_generations = 1 # extra random examples to throw in there
        self._generation_temp = 0.9
        self._generation_top_p = 0.9
        ####################################################################
        # Initialize population
        self.model = self.model.to(device)
        while len(self._genetic_pool_loss) < self._pop_size:
            input_ids = self._generate(input_ids=None)
            input_ids = tuple(input_ids.cpu().flatten().tolist())
            self._genetic_pool_loss[input_ids] = 10_000.0
            self._genetic_pool_n_correct[input_ids] = 0
        print('Original population:', self._genetic_pool_loss.keys())
    
    def _generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.generate(
            # add one to max length to account for BOS token
            input_ids=input_ids, max_length=(self._num_tokens+1), temperature=self._generation_temp, top_p=self._generation_top_p, do_sample=True
        )
    
    def _select_pop_top_k(self, k: int) -> List[Tuple[int]]:
        population = [(loss, prefix_ids) for prefix_ids, loss in self._genetic_pool_loss.items()]
        topk_pop = heapq.nsmallest(k, population)
        return [prefix_ids for _, prefix_ids in topk_pop]
    
    def _print_pop(self, top_k: int = 6) -> None:
        print((" " * 30), ("*" * 20), "Population", ("*" * 20))
        for token_ids in self._select_pop_top_k(k=top_k):
            prefix_str = "{:>70}".format(self.tokenizer.decode(list(token_ids)).replace("\n", "\\\\n"))
            loss_str = f"{self._genetic_pool_loss[token_ids]:.3f}"
            acc_str = f"{ self._genetic_pool_n_correct[token_ids]}"
            print(prefix_str, "\t\t", loss_str, "\t\t", acc_str)
        print()
    
    def _get_population(self) -> torch.Tensor:
        # TODO sample from population instead of taking top-k?
        population = self._select_pop_top_k(k=self._pop_size)
        # population_pool = self._select_pop_top_k(k=self._top_k_pop_sample)
        # population = random.sample(population_pool, self._pop_size)
        return torch.tensor(population).to(device)
    
    def _get_population_and_random_generations(self) -> torch.Tensor:
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
    
    def _mutate(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Mutates a population of prefixes.

        Truncates to a random place and then generates new options
        to try.
        """
        input_ids = input_ids.repeat((self._num_mutations_per_ex, 1))
        truncate_position = random.randint(0, self._num_tokens-1)
        if truncate_position == 0:
            return self._generate(input_ids=None)
        else:
            return self._generate(input_ids=input_ids[:, :truncate_position])
    
    def _score_population(self, population_input_ids: torch.Tensor, input_ids: torch.Tensor, next_token_ids: torch.Tensor, possible_answer_mask: torch.Tensor):
        """Scores a population of prefixes and updates `self._genetic_pool`."""
        pop_size = len(population_input_ids)
        all_candidate_losses = torch.zeros(pop_size, dtype=float).to(device)
        all_n_correct = torch.zeros(pop_size, dtype=int).to(device)
        for i in range(pop_size):
            with torch.no_grad():
                _cand_input_ids, cand_loss, cand_n_correct = (
                    self._compute_loss_with_set_prefix(
                        original_input_ids=input_ids,
                        next_token_ids=next_token_ids,
                        possible_answer_mask=possible_answer_mask,
                        prefix_ids=population_input_ids[i],
                    )
                )
            all_candidate_losses[i] += cand_loss
            all_n_correct[i] += cand_n_correct
        
        for i in range(pop_size):
            new_pop_input_ids = tuple(population_input_ids[i].cpu().tolist())
            assert len(new_pop_input_ids) == (self._num_tokens + 1) # includes BOS token
            self._genetic_pool_loss[new_pop_input_ids] = all_candidate_losses[i].item()
            self._genetic_pool_n_correct[new_pop_input_ids] = all_n_correct[i].item()

    def compute_loss_and_call_backward(
            self,
            original_input_ids: torch.Tensor,
            next_token_ids: torch.Tensor,
            possible_answer_mask: torch.Tensor
        ) -> Tuple[torch.Tensor, int]:
        """Returns the loss from the best example in the population

        Note: does not call loss.backward()
        
        Returns:
            loss (float torch.Tensor) -- the loss
            num_correct (int): number of examples where prediction was correct
        """
        self._print_pop()
        # Grab new population
        population_input_ids = self._get_population_and_random_generations()
        population_input_ids = self._mutate(population_input_ids)
        # Re-score new guys
        self._score_population(
            population_input_ids=population_input_ids,
            input_ids=original_input_ids,
            next_token_ids=next_token_ids,
            possible_answer_mask=possible_answer_mask
        )
        # Return best one
        best_prefix_ids = min(self._genetic_pool_loss, key=self._genetic_pool_loss.get)
        best_prefix_loss = self._genetic_pool_loss[best_prefix_ids]
        best_n_correct = self._genetic_pool_n_correct[best_prefix_ids]
        return best_prefix_loss, best_n_correct
        
    def post_epoch(self, dataloader: torch.utils.data.DataLoader, possible_answer_mask: torch.Tensor) -> None:
        # 
        # Get candidate IDs for every position.
        # 
        pass