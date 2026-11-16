#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 03:00:00
#SBATCH -A m1727
#SBATCH -J dbv_data
#SBATCH -o /pscratch/sd/l/laliotis/dbv/test1/logs/slurm-%j.out
# Data pipeline (pipelines/data_dp2.yaml) on one CPU node. Submit from the repository: sbatch scripts/run_data_dp2.sbatch
cd /global/cfs/cdirs/m1727/laliotis/debvader_pipeline/dbv-pipeline
scripts/warmup.sh
scripts/env_data.sh ceci pipelines/data_dp2.yaml
