#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=execute_notebook
#SBATCH --account=rwth0781
#SBATCH --output=/home/mp501934/jupyter-out/%J.out
#SBATCH --partition=c18m

#SBATCH --cpus-per-task=24
###SBATCH --mem-per-cpu=4G
#SBATCH --time=120:00:00
#SBATCH --mail-user=mareike.profe@rwth-aachen.de
#SBATCH --mail-type=END

###export PATH=~/.local/bin:$PATH

###export LD_LIBRARY_PATH="$PYTHONPATH/lib/:$LD_LIBRARY_PATH"

module load python/3.7.9 
###module load pythoni

export PATH=~/.local/bin:$PATH
export LD_LIBRARY_PATH="$PYTHONPATH/lib/:$LD_LIBRARY_PATH"

### code here
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.iopub_timeout=60 $1

