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
        # Warn (summary.yaml, log) when the stamps look bad: fraction of the in-region targets rejected by the pixel
        # cuts, and fraction of the exported objects whose shape measurement failed (NaN e1, e2, hsm_sigma).
        "warn_reject_fraction": 0.5,
        "warn_shape_fail_fraction": 0.05,
    }

    def run(self):
        from pathlib import Path
        from dp2_cutouts.stages import export_debvader
        from .utils.config import build_dp2_cutouts_config

        self.time_stamp("MakeCutouts: building the DP2-cutouts configuration")
        workdir = Path(self.config["work_dir"])/"dp2_cutouts"
        # Convert the ceci config to a DP2-cutouts config
        cfg = build_dp2_cutouts_config(self.config, workdir, label="config_MakeCutouts")
        self.time_stamp("MakeCutouts: checking the working catalogue")
        self.check_work_catalogue(cfg)

        self.STEPS = ["stamps", "empty_sky"]
        self.run_cutout_step(cfg)

        # ceci gives the in-progress paths as strings; export_* create the directories. Do not call open_output on
        # these two tags as well: for a Directory it deletes what is already there.
        iso_cutouts = Path(self.get_output("isolated_cutouts"))
        empty_cutouts = Path(self.get_output("empty_cutouts"))

        self.time_stamp("MakeCutouts: exporting the isolated dataset")
        export_debvader.export_isolated(cfg, iso_cutouts)
        self.time_stamp("MakeCutouts: exporting the empty-sky dataset")
        export_debvader.export_empty(cfg, empty_cutouts, iso_cutouts)   # stats.json is copied from the isolated set

        print(f"Cutouts written to {iso_cutouts} and {empty_cutouts}")

        qa_out = Path(self.open_output("qa_cutouts", wrapper=True).path)   # creates the directory
        self.time_stamp("MakeCutouts: cutout summary")
        self.summarise_cutouts(cfg, iso_cutouts, qa_out)
        self.time_stamp("MakeCutouts: QA montages")
        self.qa_cutouts(iso_cutouts, empty_cutouts, qa_out)
        self.time_stamp("MakeCutouts: done")

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

    def summarise_cutouts(self, cfg, isolated, qa_dir):
        """
        How many objects each pixel-level cut removes, and how many measurements failed, from the per-patch files
        of the stamps and empty_sky steps. Written to qa_dir/summary.yaml and as a funnel figure; a warning is
        printed (and recorded) when the rejected or failed fractions exceed the configured thresholds.
        """
        import json
        import numpy as np
        import pandas as pd
        from dp2_cutouts.stages.export_debvader import patch_files_dir
        from . import qa
        from .utils.misc import write_yaml

        pdir = patch_files_dir(cfg)
        t = pd.concat([pd.read_parquet(f) for f in sorted(pdir.glob("table_*.parquet"))], ignore_index=True)
        apply = list(cfg["export"]["apply"])

        def flag(name):
            col = "in_region" if name == "in_region" else f"{name}_pass"
            if col not in t:
                return np.zeros(len(t), bool)
            return t[col].astype("boolean").fillna(False).to_numpy(bool)

        def boolean(col, default=False):
            if col not in t:
                return np.full(len(t), default)
            return t[col].astype("boolean").fillna(default).to_numpy(bool)

        target = boolean("pixel_target")
        in_region = target & flag("in_region")

        # Cumulative funnel, in the order of export.apply (the export keeps the objects passing all of them).
        funnel = {"catalogue objects in the processed patches": int(len(t)),
                  "pixel targets (cut1 & cut2)": int(target.sum())}
        keep = target.copy()
        for name in apply:
            keep &= flag(name)
            funnel[f"+ {name}"] = int(keep.sum())
        n_meta = len(np.load(isolated / "meta.npz")["split"])
        if n_meta != int(keep.sum()):
            print(f"WARNING: the funnel ends at {int(keep.sum())} objects but the export holds {n_meta}")

        # Each pixel cut on its own, among the in-region targets: which cut is doing the damage.
        pixel_cuts = [n for n in apply if n not in ("cut1", "cut2", "cut6", "in_region")]
        n_in = int(in_region.sum())
        alone = {n: int((in_region & ~flag(n)).sum()) for n in pixel_cuts}

        shape_failed = in_region & ~boolean("shape_ok", default=True)
        exported_shape_failed = keep & ~np.isfinite(pd.to_numeric(t.get("e1"), errors="coerce").to_numpy(float)) \
            if "e1" in t else np.zeros(len(t), bool)
        cut8_not_evaluated = in_region & ~boolean("cut8_evaluated", default=True)

        per_patch = {}
        for (tr, pa), g in t.groupby(["tract", "patch"]):
            idx = g.index.to_numpy()
            per_patch[f"{int(tr)}/{int(pa)}"] = {"targets": int(target[idx].sum()), "in_region": int(in_region[idx].sum()),
                                               "exported": int(keep[idx].sum()),
                                               "shape_failed": int(shape_failed[idx].sum())}
        empty = {}
        for f in sorted(pdir.glob("empty_*.npz")):
            d = np.load(f)
            empty[f.stem.replace("empty_", "").replace("_", "/")] = {
                "windows": int(d["n_windows"]), "clean": int(d["n_clean"]), "written": int(len(d["ra"]))}

        def frac(a, b):
            return round(a / b, 4) if b else None

        summary = {
            "funnel": funnel,
            "in_region_targets": n_in,
            "rejected_by_each_pixel_cut_alone": {n: {"n": v, "fraction": frac(v, n_in)} for n, v in alone.items()},
            "rejected_by_pixel_cuts": {"n": n_in - int(keep.sum()), "fraction": frac(n_in - int(keep.sum()), n_in)},
            "shape_measurement_failed": {"in_region": int(shape_failed.sum()), "fraction": frac(int(shape_failed.sum()), n_in),
                                         "exported": int(exported_shape_failed.sum()),
                                         "exported_fraction": frac(int(exported_shape_failed.sum()), int(keep.sum()))},
            "cut8_not_evaluated": int(cut8_not_evaluated.sum()),
            "exported": int(keep.sum()),
            "per_patch": per_patch,
            "empty_sky_per_patch": empty,
        }
        warnings = []
        rej = summary["rejected_by_pixel_cuts"]["fraction"]
        if rej is not None and rej > self.config["warn_reject_fraction"]:
            worst = max(alone, key=alone.get) if alone else None
            warnings.append(f"the pixel cuts reject {100 * rej:.0f}% of the in-region targets "
                            f"(largest on its own: {worst}, {alone.get(worst)} objects)")
        sf = summary["shape_measurement_failed"]["exported_fraction"]
        if sf is not None and sf > self.config["warn_shape_fail_fraction"]:
            warnings.append(f"{100 * sf:.1f}% of the exported objects have no shape measurement (NaN e1, e2, hsm_sigma)")
        if int(keep.sum()) == 0:
            warnings.append("no object was exported")
        summary["warnings"] = warnings

        write_yaml(qa_dir / "summary.yaml", summary)
        print("cutout funnel:\n" + "\n".join(f"  {k:45s} {v:>8,}" for k, v in funnel.items()))
        print("rejected by each pixel cut alone (in-region targets): " + ", ".join(f"{k} {v}" for k, v in alone.items()))
        for w in warnings:
            print(f"WARNING: {w}")

        plt = qa.style()
        fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(funnel) + 0.8))
        qa.funnel(ax, list(funnel.keys()), list(funnel.values()))
        ax.set_title("Cutout funnel (cumulative, in export.apply order)")
        qa.save(fig, qa_dir, "cutout_funnel")
        return summary
