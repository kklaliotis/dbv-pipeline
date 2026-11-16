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
    }

    def run(self):
        """
        Run the PSF measurement and validation.
        """
        from .utils.misc import write_yaml

        isolated_dir = self.get_input("isolated_cutouts")
        psf_stats_out = self.get_output("psf_stats")
        qa_out = self.get_output("qa_psf")

        scale = self.config["pixel_scale"]

        fwhm = self.measure_psf_fwhm(isolated_dir, scale) 
        stats = self.compute_fwhm_stats(fwhm, self.config["bands"])
        checks = self.compare_psf_to_internal(stats, 
                                             self.config["internal_psf"], 
                                             self.config["model_pixel_scale"], 
                                             self.config["fwhm_margin"])
        
        psf_info = {"fwhm_stats": stats, 
                    "internal_psf": self.config["internal_psf"],
                    "suggested_internal_psf":
                    {band: check["suggested_internal_psf"].get(band, None) for band, check in checks.items()},
                    "ok (per band)": {band: check["ok"] for band, check in checks.items()}}
        
        write_yaml(psf_stats_out, psf_info)
        self.make_qa_figs(fwhm, checks, qa_out)

        # Read the checks and raise errors/warnings as needed
        for band, check in checks.items():
            if not check["ok"]:
                errors = [v for k, v in check.items() if k.endswith("_error")]
                raise RuntimeError("Internal PSF validation failed:\n" + "\n".join(errors))
            else:
                warnings = [v for k, v in check.items() if k.endswith("_warning")]
                if warnings:
                    print("Internal PSF validation WARNINGS:\n" + "\n".join(warnings))


    def measure_psf_fwhm(self, isolated_dir, pixel_scale):
        """
        Measure the FWHM of the PSF stamps in the isolated cutouts.

        Parameters
        ----------
        isolated_dir : DebvaderDataset
            The directory containing the isolated cutouts.
        pixel_scale : float
            The pixel scale in arcseconds/pixel.

        Returns
        -------
        fwhm : numpy.ndarray, shape (N, C)
            The measured FWHM values for each object (N) and band (C).
        """
        from dp2_cutouts.common import fwhm_from_stamp

        # load psf.npy
        psf_path = isolated_dir.path / "psf.npy"
        psf_stamps = np.load(psf_path)  # shape (N, P, P, C)

        # compute FWHM for each stamp N and each channel C
        fwhm = np.zeros((psf_stamps.shape[0], psf_stamps.shape[3]), dtype=float)
        for i in range(psf_stamps.shape[0]):
            for c in range(psf_stamps.shape[3]):
                psf_stamp = psf_stamps[i, :, :, c]
                fwhm[i, c] = fwhm_from_stamp(psf_stamp, pixel_scale)

        return fwhm

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
            A dictionary containing the FWHM percentiles for each band.
        """
        stats = {}
        for i, band in enumerate(bands):
            stats[band] = {
                "p5": np.percentile(fwhm[:, i], 5),
                "p50": np.percentile(fwhm[:, i], 50),
                "p95": np.percentile(fwhm[:, i], 95),
                "min": np.min(fwhm[:, i]),
                "max": np.max(fwhm[:, i]),
            }
        return stats

    def compare_psf_to_internal(self, stats, internal_psf, model_pixel_scale, fwhm_margin=0.9):
        """
        Compare the measured FWHM stats to the configured internal PSF. 
        Fail if:
            - G_int_band >= fwhm_min_band
            - G_int_band / model_pixel_scale < 2.0 (undersampled)
        Warn if:
            - G_int_band < fwhm_margin * fwhm_p5_band *0.6
                (G int is narrower than needed; suggest a new G_int)

        Parameters
        ----------
        stats : dict
            The FWHM percentiles for each band.
        internal_psf : dict
            The configured internal PSF parameters (G_int) per band
        model_pixel_scale : float
            The pixel scale of the model grid in arcseconds/pixel.
        fwhm_margin : float
            The margin factor for suggesting a new internal PSF.

        Returns
        -------
        check : dict
            A dictionary of dictionaries containing the validation results and suggested G_int per band.
        """
        checks = {}

        for band in stats.keys():
            check = {"ok": True, "suggested_internal_psf": {}}
            for band, band_stats in stats.items():
                fwhm_min = band_stats["min"]
                g_int = internal_psf.get("fwhm_arcsec", 0.0)

                # Check if g_int is a list (per-band) or a single float
                if isinstance(g_int, list):
                    g_int_band = g_int[stats.keys().index(band)] if 0 <= stats.keys().index(band) < len(g_int) else 0.0
                else:
                    g_int_band = g_int


                if g_int_band >= fwhm_min:
                    check["ok"] = False
                    check[f"{band}_error"] = f"Internal PSF {g_int_band:.3f} >= min observed FWHM {fwhm_min:.3f}"

                if g_int_band / model_pixel_scale < 2.0:
                    check["ok"] = False
                    check[f"{band}_error"] = f"Internal PSF {g_int_band:.3f} is undersampled"

                suggested_g_int = fwhm_margin * band_stats["p5"] * 0.6
                if g_int_band < suggested_g_int:
                    check[f"{band}_warning"] = f"Internal PSF {g_int_band:.3f} < suggested {suggested_g_int:.3f}"
                    check["suggested_internal_psf"][band] = suggested_g_int

            checks[band] = check

        return checks

    def make_qa_figs(self, fwhm, checks, qa_out):
        """
        Make QA plots for the PSF FWHM distributions. We want:
         - FWHM histogram per band with the configured and suggested G_int marked
         - PSF ellipticity histogram per band

        Parameters
        ----------
        fwhm : numpy.ndarray, shape (N, C)
            The measured FWHM values for each object (N) and band (C).
        checks : dict of dicts {band: check}
            The checks for each band, including the configured and suggested G_int values.
        qa_out : str or pathlib.Path
            The path to the output QA directory.

         """
        from .qa import make_psf_qa, save

        fig_fwhm, fig_ellip = make_psf_qa(self.config["internal_psf"], fwhm, checks)

        save(fig_fwhm, qa_out / "psf_fwhm_histograms.png")
        save(fig_ellip, qa_out / "psf_ellipticity_histograms.png")

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
