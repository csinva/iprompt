<h1 align="center">  Interpretable autoprompting </h1>
<p align="center"> Natural language explanations of a <i>dataset</i> via language-model autoprompting.
</p>

<p align="center">
  <a href="https://csinva.github.io/interpretable-autoprompting/">📚 sklearn-friendly api</a> •
  <a href="https://github.com/csinva/interpretable-autoprompting/blob/master/demo.ipynb">📖 demo notebook</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6--3.8-blue">
  <img src="https://img.shields.io/pypi/v/iprompt?color=green">
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

# Quickstart
**Installation**: `pip install embgam` (or, for more control, clone and install from source)

**Usage example** (see <a href="https://csinva.github.io/interpretable-autoprompting/">api</a> or <a href="https://github.com/csinva/interpretable-autoprompting/blob/master/demo.ipynb">demo notebook</a> for more details):

## File structure
- the main api requires simply importing `iprompt`
- the `experiments` and `experiments/scripts` folders contain hyperparameters for running sweeps contained in the paper
  - note: args that start with `use_` are boolean
- the `notebooks` folder contains notebooks for analyzing the outputs + making figures

### fMRI data experiment
- Uses scientific data/code from https://github.com/HuthLab/speechmodeltutorial linked to the paper "Natural speech reveals the semantic maps that tile human cerebral cortex" [Huth, A. G. et al., (2016) _Nature_.](https://www.nature.com/articles/nature17637)

## Testing
- to check if the pipeline seems to work, install pytest then run `pytest` from the repo's root directory


