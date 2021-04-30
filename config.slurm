#!/bin/bash
#----------------------------------------------------
# Slurm job script
#   for TACC Maverick2 GTX nodes
#----------------------------------------------------

#SBATCH -J EdgeAI-FinalProject           # Job name
#SBATCH -o EdgeAI_run%j.txt              # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e EdgeAI_run%j_error.txt        # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gtx                           # Queue (partition) name
#SBATCH -N 1                             # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                             # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00                      # Run time (hh:mm:ss)
#SBATCH --mail-user=kennethmitra@utexas.edu
#SBATCH --mail-type=all                  # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A EdgeAI-EE379K                 # Allocation name

# Other commands must follow all #SBATCH directives...

module load intel/18.0.2 python3/3.7.0
module load cuda/10.0 cudnn/7.6.2 nccl/2.4.7
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/10.0/lib64
source $WORK/final_venv/bin/activate

# Launch code...
python $WORK/final/M3_v2.py
#python $WORK/final/model_acc_tester.py
# ---------------------------------------------------

