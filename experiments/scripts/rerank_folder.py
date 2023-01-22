import itertools
import os
from os.path import join as oj
import sys
import submit_utils

save_dir = f'/home/jxm3/research/prompting/interpretable-autoprompting/results_icml/ablation_math_across_difficulty_levels'

## rerank all experiments in this folder:
# results_dir = '/home/jxm3/research/prompting/interpretable-autoprompting/results_icml/ablation3/'
# results_dir = '/home/jxm3/research/prompting/interpretable-autoprompting/results_icml/ablation2/'
results_dir = '/home/jxm3/research/prompting/interpretable-autoprompting/results_icml/ablation2_rerun/'
# results_dir = '/home/jxm3/research/prompting/interpretable-autoprompting/results_icml/ablation_math_across_difficulty_levels/'
# results_dir = '/home/jxm3/research/prompting/interpretable-autoprompting/results_icml/ablation_math_across_difficulty_levels_rerun/'

exp_dir_names = sorted(
    [oj(results_dir, fname) for fname in os.listdir(results_dir)
        if os.path.isdir(oj(results_dir, fname))
        and (
            os.path.exists(
                oj(results_dir, fname, 'results.pkl'))
        )
])
print(f'found {len(exp_dir_names)} directories with results...')

submit_utils.run_dicts(
    ["folder_name"], [[d] for d in exp_dir_names], cmd_python='python',
    script_name='03_rerank_prefix_posthoc.py', actually_run=True,
    use_slurm=True, save_dir='/home/jxm3/research/prompting/interpretable-autoprompting/slurm_files', slurm_gpu_str='gpu:1',
)
