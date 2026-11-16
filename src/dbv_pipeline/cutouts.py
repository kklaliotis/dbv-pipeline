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

        workdir = Path(self.config["work_dir"])/"dp2_cutouts"
        # Convert the ceci config to a DP2-cutouts config
        cfg = build_dp2_cutouts_config(self.config, workdir, label="config_MakeCutouts")
        self.check_work_catalogue(cfg)

        self.STEPS = ["stamps", "empty_sky"]
        self.run_cutout_step(cfg)

        # ceci gives the in-progress paths as strings; export_* create the directories. Do not call open_output on
        # these two tags as well: for a Directory it deletes what is already there.
        iso_cutouts = Path(self.get_output("isolated_cutouts"))
        empty_cutouts = Path(self.get_output("empty_cutouts"))

        export_debvader.export_isolated(cfg, iso_cutouts)
        export_debvader.export_empty(cfg, empty_cutouts, iso_cutouts)   # stats.json is copied from the isolated set

        print(f"Cutouts written to {iso_cutouts} and {empty_cutouts}")

        qa_out = Path(self.open_output("qa_cutouts", wrapper=True).path)   # creates the directory
        self.qa_cutouts(iso_cutouts, empty_cutouts, qa_out)

    def check_work_catalogue(self, cfg):
        """
        The DP2-cutouts steps read their own copy of the catalogue (<work_dir>/tables_and_stamps/catalog.parquet)
        and the other files SelectSources left in the working folder, not the selection_catalog input. Check that
        the copy exists and holds the same objects, so that the ceci dependency is real.
        """
        import pandas as pd
        from dp2_cutouts.config import TABLES_DIR

        work_cat = cfg.data(TABLES_DIR, "catalog.parquet")
        if not work_cat.exists():
            raise FileNotFoundError(
                f"{work_cat} is missing: the DP2-cutouts working folder no longer holds SelectSources' files. "
                "Delete the selection_catalog and qa_isolation outputs and re-run SelectSources.")
        ids_work = pd.read_parquet(work_cat, columns=["object_id"])["object_id"]
        ids_sel = pd.read_parquet(self.get_input("selection_catalog"), columns=["object_id"])["object_id"]
        if len(ids_work) != len(ids_sel) or set(ids_work) != set(ids_sel):
            raise RuntimeError(f"{work_cat} and the selection_catalog input hold different objects: "
                               "SelectSources was re-run without MakeCutouts' outputs being deleted, or the reverse")

    def run_cutout_step(self, cfg):
        """
        Run the two DP2-cutouts cutout steps.
        These steps write per-patch files into the DP2-cutouts working folder; export_debvader turns them into
        the two datasets.
        """
        from dp2_cutouts import cli

        for step in self.STEPS:
            self.time_stamp(f"start {step}")
            cli.run_step(cfg, step)
        self.time_stamp("cutout steps done")


    def qa_cutouts(self, isolated, empty, qa_dir):
        """
        Montages of random isolated and empty-sky stamps from the exported arrays, to check that the export
        looks like the stamps seen at the selection step and that the empty-sky stamps are indeed empty.
        """
        from .qa import make_cutout_qa, save

        bands, scale = list(self.config["bands"]), float(self.config["pixel_scale"])
        for name, path, empty_flag in (("isolated_cutouts", isolated, False), ("empty_cutouts", empty, True)):
            fig = make_cutout_qa(path, bands, scale, empty=empty_flag)
            if fig is None:
                print(f"QA montage of {name} skipped (no stamps, or no g, r, i bands)")
                continue
            save(fig, qa_dir, name)
