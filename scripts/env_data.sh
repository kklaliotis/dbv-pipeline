#!/bin/bash
# Environment for the data segment of the pipeline (pipelines/data_dp2.yaml):
# a pinned LSST stack weekly from CVMFS, plus a small CFS prefix holding the few
# packages the stack lacks (copaings, friendly, FoFCatalogMatching, huggingface_hub),
# plus the two development repositories imported straight from their source trees.
#
# Usage:   scripts/env_data.sh <command> [args...]
#   e.g.   scripts/env_data.sh ceci pipelines/data_dp2.yaml
#
# One-off install of the extras (already done on 2026-11-13):
#   scripts/env_data.sh pip install --no-deps --prefix "$DBV_DATA_PREFIX" \
#       "copaings @ git+https://github.com/LSSTDESC/copaings" \
#       "friendly @ git+https://github.com/LSSTDESC/friendly" \
#       FoFCatalogMatching fast3tree huggingface_hub hf_xet

export DBV_LSST_WEEKLY=${DBV_LSST_WEEKLY:-w_2026_46}     # pinned: change deliberately
export DBV_DATA_PREFIX=${DBV_DATA_PREFIX:-/global/cfs/cdirs/m1727/laliotis/conda/lsst-extras}
DBV_REPOS=${DBV_REPOS:-/global/cfs/cdirs/m1727/laliotis/debvader_pipeline}

source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/$DBV_LSST_WEEKLY/loadLSST-ext.bash
setup lsst_distrib
export DAF_BUTLER_REPOSITORY_INDEX=/global/cfs/cdirs/lsst/production/gen3/shared/data-repos.yaml

PYVER=$(python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH=$DBV_REPOS/dbv-pipeline/src:$DBV_REPOS/DP2-cutouts/src:$DBV_DATA_PREFIX/lib/$PYVER/site-packages:$PYTHONPATH
export PATH=$DBV_DATA_PREFIX/bin:$PATH

# Keep caches off $HOME (40 GB quota).
export HF_HOME=${HF_HOME:-$PSCRATCH/cache/huggingface}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$PSCRATCH/cache/matplotlib}
export MPLBACKEND=${MPLBACKEND:-Agg}
# Unbuffered stdout: ceci writes each stage's output to a log file, where Python would otherwise buffer prints.
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

exec "$@"
