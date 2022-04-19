#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=execute_notebook

#SBATCH --output=/work/ae664232/jupyter-out/%J.out

#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G

#SBATCH --time=72:00:00

#SBATCH -C hpcwork

export PATH=$HOME/.local/bin:$PATH
module load python/3.7.11

### code here

jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.iopub_timeout=60 $1
