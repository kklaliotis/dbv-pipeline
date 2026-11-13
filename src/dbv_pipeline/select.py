from .base_stage import PipelineStage
from .data_types import Directory, ParquetFile

# Heavy or environment-specific imports (dp2_cutouts, lsst, pandas, matplotlib) are made inside the methods:
# ceci imports every stage module in every environment.


class SelectSources(PipelineStage):
    """
    Build the catalogue of isolated galaxies that the cutouts are made from.

    The work is done by DP2-cutouts, called as a library, one step after the other:
      rubin_catalog   Rubin DP2 object table inside the sky box (Butler or TAP)
      euclid_catalog  Euclid Q1 MER catalogue + photo-z inside the box (HuggingFace)
      copaings        Rubin x Euclid matching, ellipse overlap test, blending entropy S_b
      catalog_table   one row per Rubin object, with the catalogue cuts recorded as value + pass flag

    No row is removed. The output catalogue holds every Rubin object that copaings kept (see the note below)
    together with
        cut1_pass            Euclid counterpart is neither point-like nor spurious
        cut2_pass            one-to-one Euclid match and blending entropy S_b = 0 (isolation certified by Euclid)
        rubin_deblend_pass   Rubin did not split the object from a neighbour: detect_isIsolated, or a blend
                             that produced a single child (deblend_blendNChild == 1)
        cut6_pass            i-band cModel S/N above cuts.snr_min
        selected             cut1 & cut2 & rubin_deblend & cut6: the isolated-galaxy sample at catalogue level
    The pixel-level cuts (3, 4, 5, 7, 8 and the masking decision) are applied later, by MakeCutouts.

    Note on the Rubin objects that enter: copaings keeps only objects with i_extendedness == 1 (it drops stars
    and the objects whose i-band extendedness is NaN) and with i_cModel_mag <= copaings.mag_cut. That filter is
    inside copaings.catalogs.build_lsst and is not decided here.

    Outputs
    -------
    selection_catalog : parquet, every object with its cut flags and the `selected` column
    qa_isolation      : directory with summary.yaml and the QA figures
    """

    name = "SelectSources"
    inputs = []
    outputs = [("selection_catalog", ParquetFile), ("qa_isolation", Directory)]
    config_options = {
        "work_dir": str,            # global: working area; DP2-cutouts writes under <work_dir>/dp2_cutouts
        "bands": [str],             # global (ceci needs the element type: a bare `list` breaks its parser)
        "pixel_scale": float,       # global
        "stamp_arcsec": float,      # global
        "psf_stamp_px": 35,         # global
        "split": dict,              # global
        "dp2_cutouts": dict,        # passed to DP2-cutouts (see utils.config.build_dp2_cutouts_config)
        "require_rubin_deblend": True,   # include rubin_deblend_pass in `selected`
        "montage_n": 24,            # stamps per QA montage; 0 disables the montages (they need the Butler)
        "montage_seed": 0,
    }

    STEPS = ("rubin_catalog", "euclid_catalog", "copaings", "catalog_table")

    def run(self):
        from pathlib import Path

        from .utils.config import build_dp2_cutouts_config

        # DP2-cutouts keeps all its intermediate files in one working folder (its "data_root").
        work_dir = Path(self.config["work_dir"]) / "dp2_cutouts"
        cfg = build_dp2_cutouts_config(self.config, work_dir)
        print(f"DP2-cutouts working folder: {work_dir}")

        self.run_catalogue_steps(cfg)
        catalogue = self.write_selection_catalogue(cfg)
        self.write_qa(cfg, catalogue)

    # ------------------------------------------------------------------ the three parts of run()
    def run_catalogue_steps(self, cfg):
        """Run the four DP2-cutouts catalogue steps. They write into the DP2-cutouts working folder."""
        from dp2_cutouts import cli

        for step in self.STEPS:
            self.time_stamp(f"start {step}")
            cli.run_step(cfg, step)
        self.time_stamp("catalogue steps done")

    def write_selection_catalogue(self, cfg):
        """Read the table DP2-cutouts built, add the `selected` column, write it as this stage's output."""
        import pandas as pd
        from dp2_cutouts.config import TABLES_DIR

        # <working folder>/tables_and_stamps/catalog.parquet: one row per object, written by `catalog_table`.
        catalogue = pd.read_parquet(cfg.data(TABLES_DIR, "catalog.parquet"))

        selected = catalogue["cut1_pass"] & catalogue["cut2_pass"] & catalogue["cut6_pass"]
        if self.config["require_rubin_deblend"]:
            selected &= catalogue["rubin_deblend_pass"]
        catalogue["selected"] = selected

        # ParquetFile has no writer: write to the path ceci gives (an "inprogress_" name, renamed at the end).
        path = self.get_output("selection_catalog")
        catalogue.to_parquet(path, index=False)
        print(f"selection catalogue: {len(catalogue):,} objects, {int(selected.sum()):,} selected -> {path}")
        return catalogue

    def write_qa(self, cfg, catalogue):
        """Write the summary statistics and the figures into the qa_isolation directory."""
        from pathlib import Path

        import pandas as pd
        import yaml
        from dp2_cutouts.config import EUCLID_DIR, RUBIN_DIR

        qa_dir = Path(self.open_output("qa_isolation", wrapper=True).path)   # creates the directory

        # Two catalogues the statistics need, besides the selection catalogue itself:
        # the number of Rubin objects before copaings filtered them (first line of the funnel), and the positions
        # of every Euclid source (for the distance to the nearest neighbour).
        n_rubin_objects = len(pd.read_parquet(cfg.data(RUBIN_DIR, "catalog", "object_box.parquet"),
                                              columns=["objectId"]))
        euclid_positions = pd.read_parquet(cfg.data(EUCLID_DIR, "euclid_box.parquet"), columns=["ra", "dec"])

        summary = self.summarise(catalogue, n_rubin_objects, euclid_positions, cfg)
        figures = self.make_figures(catalogue, summary, qa_dir, cfg)
        if int(self.config["montage_n"]) > 0:
            try:
                figures += self.make_montages(catalogue, qa_dir, cfg)
            except Exception as e:      # the montages are a visual check, not a product
                print(f"WARNING: montages not made ({type(e).__name__}: {e})")
                summary["montage_error"] = f"{type(e).__name__}: {e}"
        summary["figures"] = [Path(f).name for f in figures]
        with open(qa_dir / "summary.yaml", "w") as f:
            yaml.safe_dump(summary, f, sort_keys=False)
        print(yaml.safe_dump({k: summary[k] for k in ("funnel", "selected")}, sort_keys=False))
        print(f"QA: {len(figures)} figures and summary.yaml -> {qa_dir}")

    # ------------------------------------------------------------------ statistics
    @staticmethod
    def nearest_other_euclid(cat, euclid):
        """Distance [arcsec] from each Rubin object to the nearest Euclid source that is not its own counterpart.

        For an object with a one-to-one counterpart the search starts from the counterpart's position and takes
        its second neighbour (the first is itself); otherwise it starts from the Rubin position.
        """
        import numpy as np
        from scipy.spatial import cKDTree

        dec0 = float(np.median(cat["dec"]))
        cosd = np.cos(np.radians(dec0))

        def xy(ra, dec):
            return np.c_[np.asarray(ra, float) * cosd * 3600.0, np.asarray(dec, float) * 3600.0]

        tree = cKDTree(xy(euclid["ra"], euclid["dec"]))
        has = cat["has_euclid_counterpart"].to_numpy(bool) & np.isfinite(cat["euclid_ra"].to_numpy(float))
        d = np.full(len(cat), np.nan)
        if has.any():
            dd, _ = tree.query(xy(cat["euclid_ra"][has], cat["euclid_dec"][has]), k=2)
            # Guard against a counterpart position that is not exactly a tree point.
            d[has] = np.where(dd[:, 0] < 0.05, dd[:, 1], dd[:, 0])
        if (~has).any():
            dd, _ = tree.query(xy(cat["ra"][~has], cat["dec"][~has]), k=1)
            d[~has] = dd
        return d

    def summarise(self, cat, n_rubin_objects, euclid, cfg):
        import numpy as np

        sel = cat["selected"].to_numpy(bool)
        c1, c2, c6 = (cat[k].to_numpy(bool) for k in ("cut1_pass", "cut2_pass", "cut6_pass"))
        rd = cat["rubin_deblend_pass"].to_numpy(bool)
        use_rd = bool(self.config["require_rubin_deblend"])
        has = cat["has_euclid_counterpart"].to_numpy(bool)
        sb = cat["sb_euclid"].to_numpy(float)
        cat["nn_euclid_arcsec"] = self.nearest_other_euclid(cat, euclid)

        ra0, ra1, dec0, dec1 = cfg.sky_box
        area = (ra1 - ra0) * np.cos(np.radians(0.5 * (dec0 + dec1))) * (dec1 - dec0)

        def pct(x, q):
            x = np.asarray(x, float)
            x = x[np.isfinite(x)]
            return [round(float(v), 3) for v in np.percentile(x, q)] if len(x) else []

        q = [1, 5, 25, 50, 75, 95, 99]
        # Each line is the number of objects that pass this cut and all the cuts above it. The keys are the labels
        # used in the figure.
        funnel = {
            "Rubin objects in the box": int(n_rubin_objects),
            "kept by copaings (i-band extended)": int(len(cat)),
            "one-to-one Euclid counterpart": int(has.sum()),
            "cut 1: counterpart is a galaxy": int((has & c1).sum()),
            "cut 2: one-to-one and S_b = 0": int((c1 & c2).sum()),
        }
        keep = c1 & c2
        if use_rd:
            keep = keep & rd
            funnel["Rubin did not split it (isolated or 1 child)"] = int(keep.sum())
        keep = keep & c6
        funnel[f"cut 6: i-band S/N > {float(cfg['cuts']['snr_min']):g}  (selected)"] = int(keep.sum())
        summary = {
            "name": cfg.name,
            "sky_box": [float(v) for v in cfg.sky_box],
            "area_deg2": round(float(area), 5),
            "tracts": sorted(int(t) for t in np.unique(cat["tract"])),
            "patches": sorted(f"{int(t)}/{int(p)}" for t, p in
                              {(a, b) for a, b in zip(cat["tract"], cat["patch"])}),
            "rubin_source": cfg["rubin"].get("source", "tap"),
            "snr_min": float(cfg["cuts"]["snr_min"]),
            "euclid_sources_in_box": int(len(euclid)),
            "funnel": funnel,
            "blending_entropy": {
                "S_b = 0": int((sb == 0).sum()),
                "S_b > 0": int((sb > 0).sum()),
                "S_b = -1 (no Euclid overlap)": int((sb == -1).sum()),
                "S_b = 0 but not one-to-one": int(((sb == 0) & ~has).sum()),
            },
            "selected": {
                "n": int(sel.sum()),
                "fraction_of_extended": round(float(sel.mean()), 4),
                "fraction_of_extended_with_snr": round(float(sel[c6].mean()), 4) if c6.any() else None,
                "density_per_arcmin2": round(float(sel.sum() / (area * 3600.0)), 3),
                "n_per_patch": {f"{int(t)}/{int(p)}": int(n) for (t, p), n in
                                cat[sel].groupby(["tract", "patch"]).size().items()},
                "percentiles": q,
                "i_cModel_mag": pct(cat["i_cModel_mag"][sel], q),
                "i_snr": pct(cat["cut6_value"][sel], q),
                "ellipse_a_arcsec": pct(cat["ellipse_a"][sel] * float(self.config["pixel_scale"]), q),
                "resolution_i": pct(cat["resolution_i"][sel], q),
                "euclid_phz_median": pct(cat["euclid_phz_median"][sel], q),
                "nearest_other_euclid_arcsec": pct(cat["nn_euclid_arcsec"][sel], q),
                "psf_fwhm_i_arcsec (moments)": pct(cat["psf_sigma_i"][sel] * 2.3548, q),
            },
            "not_selected_extended_with_snr": {
                "n": int((~sel & c6).sum()),
                "nearest_other_euclid_arcsec": pct(cat["nn_euclid_arcsec"][~sel & c6], q),
            },
            "rubin_deblend_status": {
                "note": "objects passing cut 1, cut 2 and cut 6 (Euclid-isolated, enough signal)",
                "never deblended (detect_isIsolated)": int((c1 & c2 & c6 & cat["detect_isIsolated"].to_numpy(bool)).sum()),
                "blend with one child": int((c1 & c2 & c6 & (cat["deblend_blendNChild"].to_numpy(float) == 1)).sum()),
                "blend with 2 or more children": int((c1 & c2 & c6 & (cat["deblend_blendNChild"].to_numpy(float) >= 2)).sum()),
                "deblend flag missing": int((c1 & c2 & c6 & ~cat["detect_isIsolated"].to_numpy(bool)
                                              & ~np.isfinite(cat["deblend_blendNChild"].to_numpy(float))).sum()),
            },
        }
        return summary

    # ------------------------------------------------------------------ figures
    def make_figures(self, cat, summary, qa_dir, cfg):
        import numpy as np

        from . import qa

        plt = qa.style()
        blue, orange = qa.SERIES[:2]
        sel = cat["selected"].to_numpy(bool)
        c6 = cat["cut6_pass"].to_numpy(bool)
        parent = c6                        # Rubin galaxies above the S/N floor: what the selection draws from
        rej = parent & ~sel
        mag = cat["i_cModel_mag"].to_numpy(float)
        scale = float(self.config["pixel_scale"])
        n_sel, n_par = int(sel.sum()), int(parent.sum())
        snr = summary["snr_min"]
        figs = []

        # 1. Selection funnel.
        fig, ax = plt.subplots(figsize=(8.2, 3.4))
        qa.funnel(ax, list(summary["funnel"].keys()), list(summary["funnel"].values()))
        ax.set_title(f"Selection funnel, {summary['name']} ({summary['area_deg2']:.3f} deg$^2$)")
        figs.append(qa.save(fig, qa_dir, "01_funnel"))

        # 2. Magnitude distribution and selected fraction (two panels, one axis each).
        bins = np.arange(17.0, 28.01, 0.25)
        fig, (a0, a1) = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})
        qa.step_hist(a0, mag[parent], bins, qa.NEUTRAL, f"Rubin galaxies, S/N > {snr:g} ({n_par:,})")
        qa.step_hist(a0, mag[sel], bins, blue, f"selected, isolated ({n_sel:,})", fill=True)
        a0.set_ylabel("objects per 0.25 mag")
        a0.set_title("i-band magnitude of the selected sample")
        a0.legend(loc="upper left")
        x, f, err, _ = qa.binned_fraction(mag[parent], sel[parent], bins)
        a1.errorbar(x, f, yerr=err, color=blue, lw=2, marker="o", ms=4, elinewidth=1, capsize=0)
        a1.set_ylim(0, 1)
        a1.set_ylabel("selected fraction")
        a1.set_xlabel("i cModel magnitude (AB)")
        figs.append(qa.save(fig, qa_dir, "02_magnitude"))

        # 3. Isolation: distance to the nearest other Euclid source, selected against rejected.
        d = cat["nn_euclid_arcsec"].to_numpy(float)
        bins = np.arange(0, 12.01, 0.25)
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        qa.step_hist(ax, d[rej], bins, orange, f"not selected ({int(rej.sum()):,})", density=True)
        qa.step_hist(ax, d[sel], bins, blue, f"selected ({n_sel:,})", density=True, fill=True)
        link = float(cfg["copaings"]["linking_length"])
        ax.axvline(link, color=qa.NEUTRAL, lw=1, ls=(0, (4, 3)))
        ax.text(link + 0.1, ax.get_ylim()[1] * 0.6, "copaings linking length", color=qa.INK_2, fontsize=8)
        ax.set_xlabel("distance to the nearest other Euclid source [arcsec]")
        ax.set_ylabel("normalised density")
        ax.set_title("Isolation: nearest Euclid neighbour (any Euclid source, stars included)")
        ax.legend(loc="upper right")
        figs.append(qa.save(fig, qa_dir, "03_nearest_neighbour"))

        # 4. Blending entropy of the parent sample (left); Rubin's own deblending status (right).
        sb = cat["sb_euclid"].to_numpy(float)
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={"wspace": 0.3})
        cats = ["no Euclid\noverlap (-1)", "S_b = 0", "0 < S_b < 0.5", "S_b >= 0.5"]
        vals = [int(((sb == -1) & parent).sum()), int(((sb == 0) & parent).sum()),
                int(((sb > 0) & (sb < 0.5) & parent).sum()), int(((sb >= 0.5) & parent).sum())]
        a0.bar(range(4), vals, width=0.62, color=blue)
        a0.set_xticks(range(4), cats)
        a0.grid(axis="x", visible=False)
        for k, v in enumerate(vals):
            a0.text(k, v, f"{v:,}", ha="center", va="bottom", color=qa.INK_2, fontsize=9)
        a0.set_ylabel(f"Rubin galaxies with S/N > {snr:g}")
        a0.set_title("copaings blending entropy vs Euclid")
        # Rubin's own deblending status for objects that Euclid also calls isolated, against the rest.
        iso_flag = cat["detect_isIsolated"].to_numpy(bool)
        nchild = cat["deblend_blendNChild"].to_numpy(float)
        states = [("never deblended", iso_flag), ("blend, 1 child", ~iso_flag & (nchild == 1)),
                  ("blend, 2+ children", ~iso_flag & (nchild >= 2))]
        groups = [("Euclid-isolated (cuts 1, 2)", parent & cat["cut1_pass"].to_numpy(bool) & cat["cut2_pass"].to_numpy(bool),
                   blue),
                  ("not Euclid-isolated", parent & ~(cat["cut1_pass"].to_numpy(bool) & cat["cut2_pass"].to_numpy(bool)),
                   orange)]
        w = 0.36
        for k, (gname, gmask, color) in enumerate(groups):
            frac = [float((gmask & m).sum()) / max(int(gmask.sum()), 1) for _, m in states]
            xs = np.arange(len(states)) + (k - 0.5) * w
            a1.bar(xs, frac, width=w * 0.92, color=color, label=f"{gname} ({int(gmask.sum()):,})")
            for x_, f_ in zip(xs, frac):
                a1.text(x_, f_, f"{100 * f_:.0f}%", ha="center", va="bottom", color=qa.INK_2, fontsize=8)
        a1.set_xticks(range(len(states)), [n for n, _ in states])
        a1.grid(axis="x", visible=False)
        a1.set_ylim(0, 0.8)
        a1.set_ylabel("fraction of the group")
        a1.set_title("Rubin's own deblending status")
        a1.legend(loc="upper right")
        figs.append(qa.save(fig, qa_dir, "04_blending"))

        # 5. Sky distribution.
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        ax.scatter(cat["ra"][rej], cat["dec"][rej], s=2, color=qa.NEUTRAL, alpha=0.5, lw=0,
                   label=f"not selected ({int(rej.sum()):,})", rasterized=True)
        ax.scatter(cat["ra"][sel], cat["dec"][sel], s=4, color=blue, alpha=0.9, lw=0,
                   label=f"selected ({n_sel:,})", rasterized=True)
        ax.invert_xaxis()
        ax.set_aspect(1.0 / np.cos(np.radians(float(np.median(cat["dec"])))))
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("Dec [deg]")
        ax.set_title(f"Sky distribution (Rubin galaxies with S/N > {snr:g})")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2, markerscale=4)
        figs.append(qa.save(fig, qa_dir, "05_sky"))

        # 6. Size against magnitude: does the selection keep the size distribution of the parent?
        a_arc = cat["ellipse_a"].to_numpy(float) * scale
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw={"wspace": 0.28})
        a0.scatter(mag[rej], a_arc[rej], s=2, color=qa.NEUTRAL, alpha=0.4, lw=0, label="not selected",
                   rasterized=True)
        a0.scatter(mag[sel], a_arc[sel], s=3, color=blue, alpha=0.7, lw=0, label="selected", rasterized=True)
        a0.set_yscale("log")
        a0.set_xlim(17, 28)
        a0.set_ylim(0.2, 6)
        a0.set_xlabel("i cModel magnitude (AB)")
        a0.set_ylabel("half-light ellipse semi-major axis [arcsec]")
        a0.set_title("Size against magnitude")
        a0.legend(loc="upper right", markerscale=4)
        bins = np.logspace(np.log10(0.2), np.log10(6), 41)
        qa.step_hist(a1, a_arc[parent], bins, qa.NEUTRAL, "all with S/N cut", density=True)
        qa.step_hist(a1, a_arc[sel], bins, blue, "selected", density=True, fill=True)
        a1.set_xscale("log")
        a1.set_xlabel("semi-major axis [arcsec] (PSF included)")
        a1.set_ylabel("normalised density")
        a1.set_title("Size distribution")
        a1.legend(loc="upper right")
        figs.append(qa.save(fig, qa_dir, "06_size"))

        # 7. Redshift, S/N and PSF of the selected sample.
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), gridspec_kw={"wspace": 0.3})
        qa.step_hist(axes[0], cat["euclid_phz_median"][sel], np.arange(0, 4.01, 0.1), blue, None, fill=True)
        axes[0].set_xlabel("Euclid photo-z (median)")
        axes[0].set_ylabel("selected objects")
        axes[0].set_title("Redshift")
        qa.step_hist(axes[1], np.log10(np.clip(cat["cut6_value"][sel], 1, None)), np.arange(0.5, 3.51, 0.1),
                     blue, None, fill=True)
        axes[1].set_xlabel("log10 i-band cModel S/N")
        axes[1].set_title("Signal to noise")
        for k, b in enumerate(self.config["bands"]):
            qa.step_hist(axes[2], cat[f"psf_sigma_{b}"][sel] * 2.3548, np.arange(0.6, 2.01, 0.025),
                         qa.SERIES[k], b, lw=1.6)
        axes[2].set_xlabel("PSF FWHM from catalogue moments [arcsec]")
        axes[2].set_title("PSF size per band")
        axes[2].legend(ncols=2, loc="upper right")
        figs.append(qa.save(fig, qa_dir, "07_redshift_snr_psf"))
        return figs

    # ------------------------------------------------------------------ montages
    def make_montages(self, cat, qa_dir, cfg):
        """Colour stamps of random selected objects and of random objects rejected by the isolation cut.

        Small Butler reads (one stamp-sized box per object and band). Only with rubin.source = butler.
        """
        import numpy as np

        from . import qa
        from .utils.geometry import odd_pixels

        if cfg["rubin"].get("source", "tap") != "butler":
            print("montages skipped: they read pixels through the Butler")
            return []
        import lsst.geom as geom
        from lsst.daf.butler import Butler
        from lsst.images import Box

        b = cfg["butler"]
        butler = Butler(b["repo"], collections=b["collections"])
        skymap = butler.get("skyMap", skymap=b["skymap"], collections="skymaps")
        scale = float(self.config["pixel_scale"])
        n = odd_pixels(self.config["stamp_arcsec"], scale)
        half = n // 2
        rgb_bands = [x for x in ("i", "r", "g") if x in self.config["bands"]]
        if len(rgb_bands) != 3:
            print("montages skipped: they need the g, r and i bands")
            return []

        def stamp(row):
            wcs = skymap[int(row.tract)].getWcs()
            p = wcs.skyToPixel(geom.SpherePoint(float(row.ra), float(row.dec), geom.degrees))
            ix, iy = int(np.floor(p.x + 0.5)), int(np.floor(p.y + 0.5))
            box = Box.factory[iy - half:iy + half + 1, ix - half:ix + half + 1]
            out = []
            for band in rgb_bands:
                c = butler.get("deep_coadd", band=band, skymap=b["skymap"], tract=int(row.tract),
                               patch=int(row.patch), parameters={"bbox": box})
                out.append(np.asarray(c.image.array, float))
            return np.stack(out)

        rng = np.random.default_rng(int(self.config["montage_seed"]))
        k = int(self.config["montage_n"])
        sel = cat["selected"].to_numpy(bool)
        # Rejected by isolation only: a galaxy with enough signal that fails cut 2.
        rej = cat["cut6_pass"].to_numpy(bool) & cat["cut1_pass"].to_numpy(bool) & ~cat["cut2_pass"].to_numpy(bool)
        figs = []
        for name, mask, title in (
                ("08_montage_selected", sel, "Random selected (isolated) galaxies"),
                ("09_montage_rejected", rej, "Random galaxies rejected by the isolation cut (cut 2)")):
            idx = np.flatnonzero(mask)
            pick = np.sort(rng.choice(idx, size=min(k, len(idx)), replace=False))
            rows = cat.iloc[pick].sort_values("i_cModel_mag")
            stamps, titles = [], []
            for row in rows.itertuples():
                try:
                    stamps.append(stamp(row))
                except Exception as e:
                    print(f"  stamp of {row.object_id} not read: {type(e).__name__}: {e}")
                    stamps.append(None)
                titles.append(f"i = {row.i_cModel_mag:.1f}   S_b = {row.sb_euclid:.2g}")
            self.time_stamp(f"{name}: {sum(s is not None for s in stamps)} stamps read")
            if not any(s is not None for s in stamps):
                continue
            fig = qa.rgb_montage(stamps, titles, ncols=6, suptitle=title, pixel_scale=scale)
            figs.append(qa.save(fig, qa_dir, name))
        return figs
