from .base_stage import PipelineStage
from .data_types import DebvaderDataset, Directory, ParquetFile

# Heavy or environment-specific imports (dp2_cutouts, lsst, numpy, matplotlib) go inside run().


class MakeCutouts(PipelineStage):
    """
    Cut the stamps out of the coadds: isolated galaxies (for training) and empty sky (for noise measurement
    and injected-central blends).

    Per object and band: image, variance plane, PSF stamp and mask, plus the metadata of the dataset format
    (see docs/IMPLEMENTATION_GUIDE.md, section 4.1). The pixel-level cuts (3, 4, 5, 7, 8, the masking
    decision) are applied here, on the objects with selection_catalog["selected"] == True.

    Inputs
    ------
    selection_catalog : parquet written by SelectSources

    Outputs
    -------
    isolated_cutouts : DebvaderDataset of isolated galaxies, split by patch (train / val / test)
    empty_cutouts    : DebvaderDataset of empty-sky stamps
    qa_cutouts       : montages, masked-fraction and variance diagnostics

    STUB: nothing is implemented yet. See the guide, Phase 2 (section 8) and Phase 3 (section 9).
    """

    name = "MakeCutouts"
    inputs = [("selection_catalog", ParquetFile)]
    outputs = [("isolated_cutouts", DebvaderDataset), ("empty_cutouts", DebvaderDataset),
               ("qa_cutouts", Directory)]
    config_options = {
        "work_dir": str,            # global
        "bands": [str],             # global
        "pixel_scale": float,       # global
        "stamp_arcsec": float,      # global: stamp size; the pixel count is derived and always odd
        "psf_stamp_px": 35,         # global
        "split": dict,              # global: train / val / test fractions, assigned by patch
        "dp2_cutouts": dict,        # passed to DP2-cutouts
    }

    def run(self):
        selection_catalog = self.get_input("selection_catalog")
        isolated_out = self.get_output("isolated_cutouts")
        empty_out = self.get_output("empty_cutouts")
        qa_out = self.get_output("qa_cutouts")

        # --- planned: everything below is done by DP2-cutouts, called as a library
        # cfg = build_dp2_cutouts_config(self.config, work_dir / "dp2_cutouts")   # utils/config.py (to write)
        # for step in ("stamps", "empty_sky"):                                    # empty_sky: to write
        #     dp2_cutouts.cli.run_step(cfg, step)
        # export_isolated(cfg, isolated_out)    # dp2_cutouts.stages.export_debvader (to write)
        # export_empty(cfg, empty_out)
        # make_cutout_qa(isolated_out, empty_out, qa_out)                         # dbv_pipeline.qa
        raise NotImplementedError("MakeCutouts is a stub (guide: Phase 2 and 3)")
