#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=execute_notebook

#SBATCH --output=/work/ae664232/jupyter-out/%J.out
### SBATCH --error=/work/ae664232/jupyter-out/%J.err

#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G

#SBATCH --time=72:00:00

#SBATCH -C hpcwork


export PATH=/usr/local_rwth/sw/python/3.7.3/x86_64/bin:/usr/local_rwth/sw/python/3.7.3/x86_64/extra/bin:/opt/MPI/bin:/opt/intel/impi/2018.4.274/compilers_and_libraries/linux/mpi/bin64:/opt/intel/Compiler/19.0/1.144/rwthlnk/bin/intel64:/opt/slurm/bin:/usr/local_host/bin:/usr/local_host/sbin:/usr/local_rwth/bin:/usr/local_rwth/sbin:/usr/bin:/usr/sbin:/bin:/sbin:/opt/singularity/bin:/usr/local/bin:/usr/local/sbin:.:/home/ae664232/.local/bin:$PATH

export LD_LIBRARY_PATH=/usr/local_rwth/sw/python/3.7.3/x86_64/lib:/usr/local_rwth/sw/python/3.7.3/x86_64/extra/lib:/usr/local_rwth/sw/python/3.7.3/x86_64/extra/CBLAS:/opt/intel/impi/2018.4.274/compilers_and_libraries/linux/mpi/lib64:/opt/intel/impi/2018.4.274/compilers_and_libraries/linux/mpi/lib:/opt/intel/Compiler/19.0/1.144/rwthlnk/daal/lib/intel64_lin:/opt/intel/Compiler/19.0/1.144/rwthlnk/daal/lib/ia32_lin:/opt/intel/Compiler/19.0/1.144/rwthlnk/ipp/lib/intel64_lin:/opt/intel/Compiler/19.0/1.144/rwthlnk/ipp/lib/ia32_lin:/opt/intel/Compiler/19.0/1.144/rwthlnk/mkl/lib/intel64_lin:/opt/intel/Compiler/19.0/1.144/rwthlnk/mkl/lib/ia32_lin:/opt/intel/Compiler/19.0/1.144/rwthlnk/tbb/lib/intel64_lin/gcc4.7:/opt/intel/Compiler/19.0/1.144/rwthlnk/tbb/lib/ia32_lin/gcc4.7:/opt/intel/Compiler/19.0/1.144/rwthlnk/compiler/lib/intel64_lin:/opt/intel/Compiler/19.0/1.144/rwthlnk/compiler/lib/ia32_lin

export PYTHONPATH=/usr/local_rwth/sw/python/3.7.3/x86_64/extra/lib/python3.7/site-packages

### code here

jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.iopub_timeout=60 $1
