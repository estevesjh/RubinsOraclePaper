#!/usr/bin/env bash
#SBATCH -J nbeats-alt20
#SBATCH -A rubin:developers
#SBATCH -p milano
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH --mem=64G
#SBATCH -o logs/nbeats_alt20_%j.out
#SBATCH -e logs/nbeats_alt20_%j.err

set -euo pipefail

cd /sdf/home/e/esteves/sitcom-analysis/RubinsOraclePaper
mkdir -p logs

echo "======================================================="
echo " Re-running NBEATSx-Diff with alt_sun = -20 deg"
echo " SLURM JOB ID: ${SLURM_JOB_ID}"
echo " START: $(date)"
echo "======================================================="

# Run NBEATSx forecast
/sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python forecast/run.py

# Rebuild final results + figures
/sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python forecast/build_final.py

echo "======================================================="
echo " FINISHED at $(date)"
echo "======================================================="
