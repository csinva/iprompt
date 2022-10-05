<h1 align="center">  Interpretable autoprompting </h1>
<p align="center"> Natural language explanations of a <i>dataset</i> via language-model autoprompting.
</p>

<p align="center">
  <a href="https://csinva.github.io/interpretable-autoprompting/">📚 sklearn-friendly api</a> •
  <a href="https://github.com/csinva/interpretable-autoprompting/blob/master/demo.ipynb">📖 demo notebook</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6+-blue">
  <img src="https://img.shields.io/pypi/v/imodelsx?color=green">
</p>  


<b>Official code for using / reproducing iPrompt from the paper "Explaining Patterns in Data  with  Language Models via Interpretable Autoprompting" (<a href="https://arxiv.org/abs/2">Singh*, Morris*, Aneja, Rush & Gao, 2022</a>) </b> iPrompt generates a human-interpretable prompt that explains patterns in data while still inducing strong generalization performance.



# Quickstart
**Installation**: `pip install imodelsx` (or, for more control, clone and install from source)

**Usage example** (see <a href="https://csinva.github.io/interpretable-autoprompting/">api</a> or <a href="https://github.com/csinva/interpretable-autoprompting/blob/master/demo.ipynb">demo notebook</a> for more details):

```python
from import import Explainer
import datasets
```

# Docs
<blockquote>
<b>Abstract</b>: Large language models (LLMs) have displayed an impressive ability to harness natural language to perform complex tasks. In this work, we explore whether we can leverage this learned ability to find and explain patterns in data. Specifically, given a pre-trained LLM and data examples, we introduce interpretable autoprompting (iPrompt), an algorithm that generates a natural-language string explaining the data. iPrompt iteratively alternates between generating explanations with an LLM and reranking them based on their performance when used as a prompt. Experiments on a wide range of datasets, from synthetic mathematics to natural-language understanding, show that iPrompt can yield meaningful insights by accurately finding groundtruth dataset descriptions. Moreover, the prompts produced by iPrompt are simultaneously human-interpretable and highly effective for generalization: on real-world sentiment classification datasets, iPrompt produces prompts that match or even improve upon human-written prompts for GPT-3. Finally, experiments with an fMRI dataset show the potential for iPrompt to aid in scientific discovery.
</blockquote>

- the main api requires simply importing `imodelsx`
- the `experiments` and `experiments/scripts` folders contain hyperparameters for running sweeps contained in the paper
  - note: args that start with `use_` are boolean
- the `notebooks` folder contains notebooks for analyzing the outputs + making figures

# Related work

- **fMRI data experiment**: Uses scientific data/code from https://github.com/HuthLab/speechmodeltutorial linked to the paper "Natural speech reveals the semantic maps that tile human cerebral cortex" [Huth, A. G. et al., (2016) _Nature_.](https://www.nature.com/articles/nature17637)
- AutoPrompt: find an (uninterpretable) prompt using input-gradients ([paper](https://arxiv.org/abs/2010.15980); [github](https://github.com/ucinlp/autoprompt))
- Emb-GAM: Explain a dataset by fitting an interpretable linear model leveraging a pre-trained language model ([paper](https://arxiv.org/abs/2209.11799); [github](https://github.com/csinva/emb-gam))

## Testing
- to check if the pipeline seems to work, install pytest then run `pytest` from the repo's root directory

If this package is useful for you, please cite the following!

```r
@article{singh2022iprompt,
  year = {2022},
}
```
