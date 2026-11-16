from .base_stage import PipelineStage
from .data_types import DebvaderDataset, Directory, ParquetFile


class MakeCutouts(PipelineStage):
    """
    Cut the stamps out of the coadds: isolated galaxies (for training) and empty sky (for noise measurement
    and injected-central blends).

    Per object and band: image, variance plane, PSF stamp and mask, plus the metadata of the dataset format
    The pixel-level cuts (3, 4, 5, 7, 8, the masking decision) are applied here, on the objects with
    selection_catalog["selected"] == True.

    Inputs
    ------
    selection_catalog : parquet written by SelectSources

    Outputs
    -------
    isolated_cutouts : DebvaderDataset of isolated galaxies, split by patch (train / val / test)
    empty_cutouts    : DebvaderDataset of empty-sky stamps
    qa_cutouts       : montages, masked-fraction and variance diagnostics

    STUB: nothing is implemented yet. 
    """

    name = "MakeCutouts"
    inputs = [("selection_catalog", ParquetFile)]
    outputs = [("isolated_cutouts", DebvaderDataset), ("empty_cutouts", DebvaderDataset),
               ("qa_cutouts", Directory)]
    config_options = {
        "work_dir": str,            
        "bands": [str],             
        "pixel_scale": float,       
        "stamp_arcsec": float,      
        "psf_stamp_px": 35,         
        "split": dict,              # global: train / val / test fractions, assigned by patch
        "dp2_cutouts": dict,        # passed to DP2-cutouts
    }

    def run(self):
        from pathlib import Path
        from dp2_cutouts.stages import export_debvader
        from .utils.config import build_dp2_cutouts_config

        selection_catalog = self.get_input("selection_catalog")

        workdir = Path(self.config["work_dir"])/"dp2_cutouts"
        # Convert the ceci config to a DP2-cutouts config
        cfg = build_dp2_cutouts_config(self.config, workdir)

        self.STEPS = ["stamps", "empty_sky"]
        self.run_cutout_step(cfg)

        iso_cutouts = self.get_output("isolated_cutouts")
        empty_cutouts= self.get_output("empty_cutouts")

        export_debvader.export_isolated(cfg, iso_cutouts)
        export_debvader.export_empty(cfg, empty_cutouts)

        print(f"Cutouts written to {iso_cutouts.path} and {empty_cutouts.path}")

        qa_out = self.get_output("qa_cutouts")
        self.qa_cutouts(iso_cutouts, empty_cutouts, qa_out)

        

    def run_cutout_step(self, cfg):
        """
        Run the two DP2-cutouts cutout steps. 
        These steps write the debvader input cutouts to the work_dir and some QA outputs to the QA dir.
        The locations fo the fiels are output as DebvaderDataset and Directory objects, 
        which are passed to the next pipeline stage.
        """
        from dp2_cutouts import cli

        for step in self.STEPS:
            self.time_stamp(f"start {step}")
            cli.run_step(cfg, step)
        self.time_stamp("catalogue steps done")


    def qa_cutouts(self, isolated, empty, qa_out):
        """
        QA cutouts are already made for sources in the selection steps, but we should make sure
        that the cutouts are valid and that the empty-sky cutouts are indeed empty.

        """
        from .qa import make_cutout_qa, save

        iso_figs = []
        empty_figs = []
        qa_dir = qa_out.path

        iso_fig = make_cutout_qa(isolated, qa_out)
        iso_figs.append(save(iso_fig, qa_dir, "isolated_cutouts"))

        empty_fig = make_cutout_qa(empty, qa_out, empty=True)
        empty_figs.append(save(empty_fig, qa_dir, "empty_cutouts"))




    