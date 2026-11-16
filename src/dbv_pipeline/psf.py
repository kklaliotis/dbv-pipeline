from .base_stage import PipelineStage
from .data_types import DebvaderDataset, Directory, YamlFile
import numpy as np


class MeasurePSF(PipelineStage):
    """
    Measure the PSF width distribution per band and validate the configured internal PSF G_int.

    G_int is a configuration value (global.internal_psf). This stage:
     - checks that it is narrower than every observed PSF and well sampled
     - writes the FWHM percentiles
     - suggests a value
    Error raises if the check fails, so that nothing downstream runs with a bad G_int.

    Inputs
    ------
    isolated_cutouts : DebvaderDataset (uses psf.npy and meta.npz["psf_fwhm"])

    Outputs
    -------
    psf_stats : yaml with FWHM percentiles per band, the configured G_int, a suggested G_int, and `ok`
    qa_psf    : FWHM histograms per band with the configured and suggested G_int marked

    """

    name = "MeasurePSF"
    inputs = [("isolated_cutouts", DebvaderDataset)]
    outputs = [("psf_stats", YamlFile), ("qa_psf", Directory)]
    config_options = {
        "bands": [str],             # global
        "pixel_scale": float,       # global: data grid, arcsec per pixel
        "model_pixel_scale": float,  # global: decoder grid (equal to pixel_scale in Test I)
        "internal_psf": dict,       # global: {type, beta, fwhm_arcsec}
        "fwhm_margin": 0.9,         # suggest G_int = margin * (narrowest observed FWHM)
        "min_sampling_px": 2.5,     # G_int FWHM must cover at least this many model pixels (guide, section 3.4)
    }

    def run(self):
        """
        Run the PSF measurement and validation.
        """
        from pathlib import Path
        from .utils.misc import write_yaml

        # ceci gives paths as strings; the Directory output must be created (open_output does that).
        isolated_dir = Path(self.get_input("isolated_cutouts"))
        psf_stats_out = self.get_output("psf_stats")
        qa_out = Path(self.open_output("qa_psf", wrapper=True).path)

        bands = list(self.config["bands"])
        scale = self.config["pixel_scale"]

        fwhm = self.read_psf_fwhm(isolated_dir, bands)
        ellipticity = self.measure_psf_ellipticity(isolated_dir, fwhm / scale)
        stats = self.compute_fwhm_stats(fwhm, bands)
        checks = self.compare_psf_to_internal(stats,
                                              self.config["internal_psf"],
                                              self.config["model_pixel_scale"],
                                              self.config["fwhm_margin"],
                                              self.config["min_sampling_px"])
        ok = all(check["ok"] for check in checks.values())

        psf_info = {"ok": ok,
                    "n_objects": int(fwhm.shape[0]),
                    "fwhm_stats": stats,
                    "internal_psf": self.config["internal_psf"],
                    "suggested_internal_psf_fwhm": {band: check["suggested_fwhm"] for band, check in checks.items()},
                    "ok_per_band": {band: check["ok"] for band, check in checks.items()},
                    "errors": [e for check in checks.values() for e in check["errors"]],
                    "warnings": [w for check in checks.values() for w in check["warnings"]]}

        write_yaml(psf_stats_out, psf_info)
        self.make_qa_figs(fwhm, ellipticity, checks, bands, qa_out)

        # Read the checks and raise errors/warnings as needed (after the files are written, so they can be read)
        if psf_info["warnings"]:
            print("Internal PSF validation WARNINGS:\n" + "\n".join(psf_info["warnings"]))
        if not ok:
            raise RuntimeError("Internal PSF validation failed:\n" + "\n".join(psf_info["errors"]))


    def read_psf_fwhm(self, isolated_dir, bands):
        """
        Read the FWHM of the PSF stamps of the isolated cutouts from meta.npz["psf_fwhm"].

        DP2-cutouts (stamps.py) measures it with dp2_cutouts.common.psf.fwhm_from_stamp on the same float32 PSF
        stamp that is exported to psf.npy, so measuring it again here would repeat the same computation.

        Parameters
        ----------
        isolated_dir : pathlib.Path
            The DebvaderDataset directory containing the isolated cutouts.
        bands : list of str
            The configured bands, checked against the dataset's.

        Returns
        -------
        fwhm : numpy.ndarray, shape (N, C)
            The FWHM in arcsec for each object (N) and band (C).
        """
        meta = np.load(isolated_dir / "meta.npz")
        if "bands" in meta and list(meta["bands"]) != list(bands):
            raise ValueError(f"{isolated_dir} has bands {list(meta['bands'])}, the configuration {list(bands)}")
        fwhm = np.asarray(meta["psf_fwhm"], dtype=float)  # shape (N, C)
        if fwhm.ndim != 2 or fwhm.shape[1] != len(bands):
            raise ValueError(f"meta.npz['psf_fwhm'] has shape {fwhm.shape}, expected (N, {len(bands)})")
        if not np.isfinite(fwhm).all():
            raise ValueError(f"{int((~np.isfinite(fwhm)).sum())} non-finite PSF FWHM values in {isolated_dir}: "
                             "objects without a PSF should have been removed by the `psf` export cut")
        return fwhm

    def measure_psf_ellipticity(self, isolated_dir, fwhm_px, chunk=2000):
        """
        Ellipticity |e| = sqrt(e1^2 + e2^2), e1 = (Ixx - Iyy) / (Ixx + Iyy), e2 = 2 Ixy / (Ixx + Iyy), of each
        PSF stamp, from second moments about the stamp centre (P // 2) with a circular Gaussian weight of the
        stamp's own FWHM. The weight biases |e| low compared with adaptive moments, but it is the same for every
        stamp, so the histogram shows the spread and the outliers, which is what the QA is for.

        Parameters
        ----------
        isolated_dir : pathlib.Path
            The DebvaderDataset directory containing the isolated cutouts.
        fwhm_px : numpy.ndarray, shape (N, C)
            The FWHM of each stamp, in pixels (sets the weight).

        Returns
        -------
        ellipticity : numpy.ndarray, shape (N, C)
        """
        psf_stamps = np.load(isolated_dir / "psf.npy", mmap_mode="r")  # (N, P, P, C)
        n, size = psf_stamps.shape[0], psf_stamps.shape[1]
        y, x = np.mgrid[:size, :size] - size // 2
        x, y = x[None, :, :, None], y[None, :, :, None]
        out = np.full((n, psf_stamps.shape[3]), np.nan)
        for a in range(0, n, chunk):
            p = np.asarray(psf_stamps[a:a + chunk], dtype=float)
            sigma = (fwhm_px[a:a + chunk] / 2.3548)[:, None, None, :]
            w = p * np.exp(-0.5 * (x * x + y * y) / sigma**2)
            ixx, iyy, ixy = ((w * q).sum(axis=(1, 2)) for q in (x * x, y * y, x * y))
            with np.errstate(invalid="ignore", divide="ignore"):
                out[a:a + chunk] = np.hypot(ixx - iyy, 2.0 * ixy) / (ixx + iyy)
        return out

    def compute_fwhm_stats(self, fwhm, bands):
        """
        Compute the FWHM percentiles for each band.

        Parameters
        ----------
        fwhm : numpy.ndarray
            The measured FWHM values for each object and band.
        bands : list of str
            The list of band names corresponding to the columns in `fwhm`.

        Returns
        -------
        stats : dict
            A dictionary containing the FWHM percentiles for each band (plain floats, so they can go into yaml).
        """
        stats = {}
        for i, band in enumerate(bands):
            stats[band] = {
                "min": float(np.min(fwhm[:, i])),
                "p1": float(np.percentile(fwhm[:, i], 1)),
                "p5": float(np.percentile(fwhm[:, i], 5)),
                "p50": float(np.percentile(fwhm[:, i], 50)),
                "p95": float(np.percentile(fwhm[:, i], 95)),
                "max": float(np.max(fwhm[:, i])),
            }
        return stats

    def compare_psf_to_internal(self, stats, internal_psf, model_pixel_scale, fwhm_margin=0.9, min_sampling_px=2.5):
        """
        Compare the measured FWHM stats to the configured internal PSF.
        Fail if:
            - G_int_band >= fwhm_min_band
            - G_int_band / model_pixel_scale < min_sampling_px (undersampled)
        Warn if:
            - G_int_band < fwhm_margin * fwhm_p1_band * 0.6
                (G int is narrower than needed)
        Suggest, for every band: max(min_sampling_px * model_pixel_scale, fwhm_margin * fwhm_min_band)

        Parameters
        ----------
        stats : dict
            The FWHM percentiles for each band.
        internal_psf : dict
            The configured internal PSF parameters (G_int); fwhm_arcsec is one number or one per band
        model_pixel_scale : float
            The pixel scale of the model grid in arcseconds/pixel.
        fwhm_margin : float
            The margin factor for suggesting a new internal PSF.
        min_sampling_px : float
            Minimum G_int FWHM in model pixels. At 2.0 px about 6% of the Moffat's Fourier amplitude is left at
            the pixel Nyquist frequency (aliased); at 2.5 px about 1.4%.

        Returns
        -------
        checks : dict
            {band: {"ok", "fwhm", "suggested_fwhm", "errors", "warnings"}}
        """
        bands = list(stats.keys())
        g_int = internal_psf.get("fwhm_arcsec")
        if g_int is None:
            raise KeyError("internal_psf.fwhm_arcsec is not set")
        if isinstance(g_int, (list, tuple)):
            if len(g_int) != len(bands):
                raise ValueError(f"internal_psf.fwhm_arcsec has {len(g_int)} values for {len(bands)} bands")
            g_int_per_band = dict(zip(bands, (float(v) for v in g_int)))
        else:
            g_int_per_band = {band: float(g_int) for band in bands}

        checks = {}
        for band, band_stats in stats.items():
            g_int_band = g_int_per_band[band]
            fwhm_min = band_stats["min"]
            check = {"ok": True, "fwhm": g_int_band, "errors": [], "warnings": [],
                     "suggested_fwhm": max(min_sampling_px * model_pixel_scale, fwhm_margin * fwhm_min)}

            if g_int_band >= fwhm_min:
                check["ok"] = False
                check["errors"].append(f"{band}: internal PSF {g_int_band:.3f} >= min observed FWHM {fwhm_min:.3f}")

            if g_int_band / model_pixel_scale < min_sampling_px - 1e-9:     # tolerance: 0.5 / 0.2 in floating point
                check["ok"] = False
                check["errors"].append(f"{band}: internal PSF {g_int_band:.3f} is undersampled "
                                       f"({g_int_band / model_pixel_scale:.2f} < {min_sampling_px} model pixels)")

            narrow = fwhm_margin * band_stats["p1"] * 0.6
            if g_int_band < narrow:
                check["warnings"].append(f"{band}: internal PSF {g_int_band:.3f} < {narrow:.3f} (much narrower "
                                         f"than needed; suggested {check['suggested_fwhm']:.3f})")

            checks[band] = check

        return checks

    def make_qa_figs(self, fwhm, ellipticity, checks, bands, qa_out):
        """
        Make QA plots for the PSF FWHM distributions. We want:
         - FWHM histogram per band with the configured and suggested G_int marked
         - PSF ellipticity histogram per band

        Parameters
        ----------
        fwhm : numpy.ndarray, shape (N, C)
            The measured FWHM values for each object (N) and band (C).
        ellipticity : numpy.ndarray, shape (N, C)
            The PSF ellipticity of each object and band.
        checks : dict of dicts {band: check}
            The checks for each band, including the configured and suggested G_int values.
        bands : list of str
        qa_out : pathlib.Path
            The path to the output QA directory.

         """
        from .qa import make_psf_qa, save

        fig_fwhm, fig_ellip = make_psf_qa(fwhm, ellipticity, checks, bands)

        save(fig_fwhm, qa_out, "psf_fwhm_histograms")
        save(fig_ellip, qa_out, "psf_ellipticity_histograms")

        return None



class CheckKernels(PipelineStage):
    """
    Check the matching kernel K, with K * G_int = P, for every object and band.

    The kernel itself is computed later, in the data loader, from the same debvader function. This stage only
    asserts that max |K~| <= kernel_max and that K * G_int reproduces P to kernel_recon_tol, and records the
    G_int that was validated so that training can refuse a different one.

    Inputs
    ------
    isolated_cutouts : DebvaderDataset
    psf_stats        : yaml from MeasurePSF (refuses to run if `ok` is false)

    Outputs
    -------
    kernel_report : yaml with the per-band maxima, the internal_psf block that was used, and failing object ids
    qa_kernels    : histograms of max |K~| and of the reconstruction error; montage of the worst objects

    STUB: nothing is implemented yet. See the guide, section 10.3 (and 11.3 for the kernel code).
    """

    name = "CheckKernels"
    inputs = [("isolated_cutouts", DebvaderDataset), ("psf_stats", YamlFile)]
    outputs = [("kernel_report", YamlFile), ("qa_kernels", Directory)]
    config_options = {
        "internal_psf": dict,       # global
        "kernel_reg": float,        # global: Tikhonov regularisation of the division by G~
        "kernel_max": float,        # global: assert max |K~| <= kernel_max
        "kernel_recon_tol": float,  # global: assert max|K*G - P| / max P <= tol
        "max_objects": 0,           # 0 = every object
    }

    def run(self):
        isolated_dir = self.get_input("isolated_cutouts")
        psf_stats = self.get_input("psf_stats")
        kernel_report_out = self.get_output("kernel_report")
        qa_out = self.get_output("qa_kernels")

        # --- planned (debvader.physics.forward, guide section 11.3: to write)
        # g_int = internal_psf_stamp(psf_px, fwhm_pix, **self.config["internal_psf"])      # per band
        # for each object and band:
        #     kernel, abs_k = matching_kernel(psf, g_int, self.config["kernel_reg"])
        #     record max(abs_k) and max|convolve(g_int, kernel) - psf| / max(psf)
        # report = assert_kernels_ok(records, self.config["kernel_max"], self.config["kernel_recon_tol"])
        # write_yaml(kernel_report_out, report)
        # make_kernel_qa(records, qa_out)
        raise NotImplementedError("CheckKernels is a stub (guide: section 10.3)")
