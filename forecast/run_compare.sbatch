#!/usr/bin/env bash
#SBATCH --job-name=nbx-cmp
#SBATCH --account=m1727_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/cmp_%j.out
#SBATCH --error=logs/cmp_%j.err

set -euo pipefail

cd /pscratch/sd/j/jesteves/github/RubinsOraclePaper
mkdir -p results logs

module load pytorch/2.8.0
export PYTHONUSERBASE=$PSCRATCH/python-userbase/pytorch2.8.0
export HDF5_USE_FILE_LOCKING=FALSE
export KMP_DUPLICATE_LIB_OK=TRUE
export RUN_ACCEL=gpu

PYBIN=/global/common/software/nersc9/pytorch/2.8.0/bin/python3
${PYBIN} -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

# Driven by env: RUN_FLAGS="" for no-MB, RUN_FLAGS="--mb" for with-MB
RUN_FLAGS=${RUN_FLAGS:-}
echo "=== run.py ${RUN_FLAGS} on $(hostname) ==="
${PYBIN} forecast/run.py ${RUN_FLAGS}
echo "=== done $(date) ==="
