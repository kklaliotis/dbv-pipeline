from .base_stage import PipelineStage
from .dummy_stage import DummyStage

# Importing the stage modules registers the stages with ceci. This is required: ceci launches a stage as
# `python -m dbv_pipeline <StageName>`, and that process only knows the stages this package imports
# (the `modules:` list of the pipeline file is not used there).
# Keep every heavy or environment-specific import (lsst, keras, debvader, dp2_cutouts) inside run().
from . import select, cutouts, psf, noise  # noqa: E402,F401
