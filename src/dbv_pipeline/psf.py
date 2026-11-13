from .base_stage import PipelineStage
from .data_types import DebvaderDataset, Directory, YamlFile

# Heavy or environment-specific imports (numpy, scipy, debvader) go inside run().


class MeasurePSF(PipelineStage):
    """
    Measure the PSF width distribution per band and validate the configured internal PSF G_int.

    G_int is a configuration value (global.internal_psf), never chosen here. This stage checks that it is
    narrower than every observed PSF and well sampled, writes the FWHM percentiles, and suggests a value. It
    raises if the check fails, so that nothing downstream runs with a bad G_int.

    Inputs
    ------
    isolated_cutouts : DebvaderDataset (uses psf.npy and meta.npz["psf_fwhm"])

    Outputs
    -------
    psf_stats : yaml with FWHM percentiles per band, the configured G_int, a suggested G_int, and `ok`
    qa_psf    : FWHM histograms per band with the configured and suggested G_int marked

    STUB: nothing is implemented yet. See the guide, section 10.1.
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
        isolated_dir = self.get_input("isolated_cutouts")
        psf_stats_out = self.get_output("psf_stats")
        qa_out = self.get_output("qa_psf")

        # --- planned
        # fwhm = measure_psf_fwhm(isolated_dir, self.config["pixel_scale"])         # per object and band
        # stats = summarise_fwhm(fwhm, self.config["bands"])                        # percentiles per band
        # check = validate_internal_psf(stats, self.config["internal_psf"],
        #                               self.config["model_pixel_scale"], self.config["fwhm_margin"])
        # write_yaml(psf_stats_out, stats, check)
        # make_psf_qa(fwhm, check, qa_out)                                          # dbv_pipeline.qa
        # if not check["ok"]: raise RuntimeError("internal PSF is not narrower than every observed PSF")
        raise NotImplementedError("MeasurePSF is a stub (guide: section 10.1)")


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
