#!/bin/bash
#SBATCH -J {job_name}                        # Job name
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -p rush-interactive                              # Total number of nodes requested
#SBATCH -n {num_cpus}                    # Total number of cores requested
#SBATCH --mem={mem}                      # Total amount of (real) memory requested (per node)
#SBATCH --gres={gpu}
#SBATCH -t 48:00:00                          # Time limit (hh:mm:ss)
#SBATCH --requeue                            # Put back in the queue if we get preempted
#SBATCH --constraint="gpu-mid|gpu-high"
#SBATCH --output={save_dir}/R-%x.%j.out                 # Output file. (%j gives job ID, %x gives job name)

# source /home/jxm3/.zshrc # <-- won't work bc SLURM runs in batch

# cd random/interpretable_autoprompting


# TODO email self when job finished :-)
{python_cmd}