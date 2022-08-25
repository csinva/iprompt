# On Robust Prefix-Tuning for Text Classification

Prefix-tuning has drawn much attention as it is a parameter-efficient and modular alternative to adapting pretrained language models to downstream tasks. However, we find that prefix-tuning suffers from adversarial attacks. While, unfortunately, current robust NLP methods are unsuitable for prefix-tuning as they will inevitably hamper the modularity of prefix-tuning. In our ICLR'22 [paper](https://openreview.net/forum?id=eBCmOocUejf), we propose robust prefix-tuning for text classification. Our method leverages the idea of test-time tuning, which preserves the strengths of prefix-tuning and improves its robustness at the same time. This repository contains the code for the proposed robust prefix-tuning method.

## Prerequisite

PyTorch>=1.2.0, pytorch-transformers==1.2.0, OpenAttack==2.0.1, and GPUtil==1.4.0. 

## Train the original prefix P_θ

For the training phase of standard prefix-tuning, the command is:

```bash
  source train.sh --preseqlen [A] --learning_rate [B] --tasks [C] --n_train_epochs [D] --device [E]
```

where
- [A]: The length of the prefix P_θ.
- [B]: The (initial) learning rate. 
- [C]: The benchmark. Default: sst. 
- [D]: The total epochs during training.
- [E]: The id of the GPU to be used.

We can also use adversarial training to improve the robustness of the prefix. For the training phase of adversarial prefix-tuning, the command is:

```bash
  source train_adv.sh --preseqlen [A] --learning_rate [B] --tasks [C] --n_train_epochs [D] --device [E] --pgd_ball [F]
```

where
- [A]~[E] have the same meanings with above.
- [F]: where norm ball is word-wise or sentence-wise.

Note that the DATA_DIR and MODEL_DIR in train_adv.sh are different from those in train.sh. When experimenting with the adversarially trained prefix P_θ's in the following steps, remember to switch the DATA_DIR and MODEL_DIR in the corresponding scripts as well.

## Generate Adversarial Examples 

We use the OpenAttack package to generate in-sentence adversaries. The command is:

```bash
  source generate_adv_insent.sh --preseqlen [A] --learning_rate [B] --tasks [C] --device [E] --test_ep [G] --attack [H]
```

where
- [A],[B],[C],[E] have the same meanings with above.
- [G]: Load the prefix P_θ parameters trained for [G] epochs for testing. We set G=D.
- [H]: Generate adversarial examples based on clean test set with the in-sentence attack [H].

We also implement the Universal Adversarial Trigger attack. The command is:

```bash
  source generate_adv_uat.sh --preseqlen [A] --learning_rate [B] --tasks [C] --device [E] --test_ep [G] --attack clean-[H2] --uat_len [I] --uat_epoch [J]
```

where
- [A],[B],[C],[E],[G] have the same meanings with above.
- \[H2]: We should search for UATs for each class in the benchmark, and H2 indicates the class id. H2=0/1 for SST, 0/1/2/3 for AG News, and 0/1/2 for SNLI.
- \[I]: The length of the UAT.
- \[J]: The epochs for exploiting UAT.

## Test the performance of P_θ

The command for performance testing of P_θ under clean data and in-sentence attacks is:

```bash
  source test_prefix_theta_insent.sh --preseqlen [A] --learning_rate [B] --tasks [C] --device [E] --test_ep [G] --attack [H] --test_batch_size [K]
```

Under UAT attack, the test command is:

```bash
  source test_prefix_theta_uat.sh --preseqlen [A] --learning_rate [B] --tasks [C] --device [E] --test_ep [G] --attack clean --uat_len [I] --test_batch_size [K]
```

where
- [A]~[I] have the same meanings with above.
- [K]: The test batch size. when K=0, the batch size is adaptive (determined by GPU memory); when K>0, the batch size is fixed.

## Robust Prefix P'_ψ: Constructing the canonical manifolds

By constructing the canonical manifolds with PCA, we get the projection matrices. The command is:

```bash
  source get_proj.sh --preseqlen [A] --learning_rate [B] --tasks [C] --device [E] --test_ep [G]
```

where [A]~[G] have the same meanings with above.  

## Robust Prefix P'_ψ: Test its performance

Under clean data and in-sentence attacks, the command is:

```bash
  source test_robust_prefix_psi_insent.sh --preseqlen [A] --learning_rate [B] --tasks [C] --device [E] --test_ep [G] --attack [H] --test_batch_size [K] --PMP_lr [L] --PMP_iter [M]
```

Under UAT attack, the test command is:

```bash
  source test_robust_prefix_psi_uat.sh --preseqlen [A] --learning_rate [B] --tasks [C] --device [E] --test_ep [G] --attack clean --uat_len [I] --test_batch_size [K] --PMP_lr [L] --PMP_iter [M]
```

where 
- [A]~[K] have the same meanings with above.  
- \[L]: The learning rate for test-time P'\_ψ tuning.
- \[M]: The iterations for test-time P'_ψ tuning.

## Running Example

```bash
# Train the original prefix P_θ
source train.sh --tasks sst --n_train_epochs 100 --device 0
source train_adv.sh --tasks sst --n_train_epochs 100 --device 1 --pgd_ball word

# Generate Adversarial Examples
source generate_adv_insent.sh --tasks sst --device 0 --test_ep 100 --attack bug
source generate_adv_uat.sh --tasks sst --device 0 --test_ep 100 --attack clean-0 --uat_len 3 --uat_epoch 10
source generate_adv_uat.sh --tasks sst --device 0 --test_ep 100 --attack clean-1 --uat_len 3 --uat_epoch 10

# Test the performance of P_θ
source test_prefix_theta_insent.sh --tasks sst --device 0 --test_ep 100 --attack bug --test_batch_size 0
source test_prefix_theta_uat.sh --tasks sst --device 0 --test_ep 100 --attack clean --uat_len 3 --test_batch_size 0

# Robust Prefix P'_ψ: Constructing the canonical manifolds
source get_proj.sh --tasks sst --device 0 --test_ep 100

# Robust Prefix P'_ψ: Test its performance
source test_robust_prefix_psi_insent.sh --tasks sst --device 0 --test_ep 100 --attack bug --test_batch_size 0 --PMP_lr 0.15 --PMP_iter 10
source test_robust_prefix_psi_uat.sh --tasks sst --device 0 --test_ep 100 --attack clean --uat_len 3 --test_batch_size 0 --PMP_lr 0.05 --PMP_iter 10

```

## Released Data & Models

The training the original prefix P\_θ and the process of generating adversarial examples can be time-consuming. As shown in our [paper](https://openreview.net/forum?id=eBCmOocUejf), the adversarial prefix-tuning is particularly slow. Efforts need to be paid on generating adversaries as well, since different attacks are to be performed on the test set based on each trained prefix. We also found that OpenAttack is now upgraded to v2.1.1, which causes compatibility issues in our codes (test_prefix_theta_insent.py).

In order to facilitate research on the robustness of prefix-tuning, we release the prefix checkpoints P_θ (with both std. and adv. training), the processed test sets that are perturbed by in-sentence attacks (including PWWS and TextBugger), as well as the generated projection matrices of the canonical manifolds in our runs for reproducibility and further enhancement. We have also hard-coded the exploited UAT tokens in test_prefix_theta_uat.py and test_robust_prefix_psi_uat.py. All the materials can be found [here](https://drive.google.com/file/d/1842WbRH6y3TYuXuURfFIIvm4w6mTqYXT/view?usp=sharing).


## Acknowledgements:
The implementation of robust prefix tuning is based on the [LAMOL](https://github.com/chho33/LAMOL) repo, which is the code of [LAMOL: LAnguage MOdeling for Lifelong Language Learning](https://openreview.net/forum?id=Skgxcn4YDS) that studies NLP lifelong learning with GPT-style pretrained language models.

## Bibtex

If you find this repository useful for your research, please consider citing our work:

```
@inproceedings{
  yang2022on,
  title={On Robust Prefix-Tuning for Text Classification},
  author={Zonghan Yang and Yang Liu},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=eBCmOocUejf}
}
```
