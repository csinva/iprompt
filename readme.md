<h1 align="center">  Interpretable autoprompting </h1>
<p align="center"> Natural language explanations of a <i>dataset</i> via language-model autoprompting.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6--3.8-blue">
</p>  


<b>Official code for using / reproducing Interpretable autoprompting from the paper "Towards scientific discovery with language models via interpretable autoprompting" (<a href="https://arxiv.org/abs/2">Singh*, Morris*, Aneja, Rush & Gao, 2022</a>) </b>

<blockquote>
<b>Abstract</b>: Large language models (LLMs) have displayed an extraordinary ability to harness natural language and perform complex tasks.
In this work, we explore whether we can leverage a pre-trained LLM to understand patterns in our data.
This ambitious problem statement has the potential to fuel scientific discovery, especially if an LLM is able to identify and explain structure in data that elude humans.
Our approach to this problem is grounded in automatic prompt tuning:
it iterates over the entire dataset, queries the model, and then uses the aggregate results to derive a natural-language interpretation.
Experiments on a wide range of tasks, ranging from synthetic mathematics to diverse natural-language-understanding tasks show that this problem statement is feasible.
</blockquote>

## Setup
- `pip install -r requirements.txt`

## File structure
- `XX_train_XX.py` files each launch a job to fit a different task
  - a lot of the main modeling code is in `model_utils`
- `scripts` is a folder for running sweeps over experiments
  - for example, `scripts/submit_sweep.py` loops over cmd-line args and calls `01_train.py`
- `notebooks` folder contains notebooks for analyzing results: training scripts save a pkl of results into a folder and after the sweep is run the `analyze` notebooks load and aggregate these into a dataframe
- `data.py` holds the code for generating datasets
  - it uses files in the `data_utils` folder


## fMRI data experiment
- Uses scientific data/code from https://github.com/HuthLab/speechmodeltutorial linked to the paper "Natural speech reveals the semantic maps that tile human cerebral cortex" [Huth, A. G. et al., (2016) _Nature_.](https://www.nature.com/articles/nature17637)

## Testing
- to check if the pipeline seems to work, install pytest then run `pytest` from the repo's root directory

## Notes
- note: args that start with `use_` are boolean
