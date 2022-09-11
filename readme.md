<h1 align="center">   <img src="https://yu-group.github.io/adaptive-wavelets/anim.gif" width="15%"> Interpretable autoprompting <img src="https://yu-group.github.io/adaptive-wavelets/anim.gif" width="15%"></h1>
<p align="center"> Natural language explanations of a <i>dataset</i> via language-model autoprompting.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6--3.8-blue">
  <a href="https://github.com/csinva/imodels/actions"><img src="https://github.com/Yu-Group/adaptive-wavelets/workflows/tests/badge.svg"></a>
</p>  


<b>Official code for using / reproducing Interpretable autoprompting from the paper "Towards scientific discovery with language models via interpretable autoprompting" (<a href="https://arxiv.org/abs/2">(Singh, Morris, Aneja, Rush & Gao, 2022</a>)

<blockquote>
<b>Abstract</b>: Large language models (LLMs) have displayed an extraordinary ability to harness natural language and perform complex tasks.
In this work, we explore whether we can leverage a pre-trained LLM to understand patterns in our data.
This ambitious problem statement has the potential to fuel scientific discovery, especially if an LLM is able to identify and explain structure in data that elude humans.
Our approach to this problem is grounded in automatic prompt tuning:
it iterates over the entire dataset, queries the model, and then uses the aggregate results to derive a natural-language interpretation.
Experiments on a wide range of tasks, ranging from synthetic mathematics to diverse natural-language-understanding tasks show that this problem statement is feasible.
</blockquote>


## file structure
- `01_train_suffix.py` is the main function to run and deals with processing all the cmd-line args
  - *suffix* is much simpler (and doesn't require any model gradients)
  - when we implement *classification* and *clustering*, these should also go into new, different files
- `scripts` is a folder for running sweeps over experiments
  - for example, `scripts/submit_sweep.py` loops over cmd-line args and calls `01_train.py`
  - each file saves a pkl of results into a folder and after the sweep is run the `analyze` notebooks load and aggregate these into a dataframe
- `data.py` holds the code for generating datasets
  - it uses files in the `data_utils` folder

## testing
- to check if the pipeline seems to work, install pytest then run `pytest` from the repo's root directory

## docs
- note: args that start with `use_` are boolean