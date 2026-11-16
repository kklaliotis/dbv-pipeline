#!/bin/bash
# Import the heavy packages of the data pipeline once, timed, before running ceci.
# On a fresh compute node the first import of the LSST stack (CVMFS) and of the code on CFS can take minutes;
# doing it here keeps that cost out of the stage timings and makes it visible instead of looking like a hang.
# Usage: scripts/warmup.sh        (it sources the data environment itself)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo "[$(date '+%F %T')] warm-up on $(hostname): importing the data-pipeline packages"
time "$HERE/env_data.sh" python -c "
import lsst.daf.butler, lsst.geom, lsst.images
import numpy, scipy.ndimage, pandas, pyarrow.parquet, astropy.wcs, galsim, healpy, lsdb
import ceci, dbv_pipeline
import dp2_cutouts.cli, dp2_cutouts.stages.stamps, dp2_cutouts.stages.empty_sky, dp2_cutouts.stages.export_debvader
"
echo "[$(date '+%F %T')] warm-up done"
