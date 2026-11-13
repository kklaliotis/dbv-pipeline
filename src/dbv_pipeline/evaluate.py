from .base_stage import PipelineStage
from .data_types import BlendDataset, Directory, HDFFile, YamlFile

# Heavy or environment-specific imports (numpy, galsim, matplotlib) go inside run().


class Evaluate(PipelineStage):
    """
    Measure how well the deblender recovers the central galaxy, against truth and against no deblending.

    Per scene: flux and colour residuals, HSM size and shape residuals, for the model at the observed PSF and,
    as a baseline, for the blend itself. Bias and scatter are reported in bins of multiplicity, magnitude
    difference, separation, blendedness, S/N, PSF width and sub-pixel offset, separately for the real-central
    and injected-central sets, and for contaminants brighter than the central object.

    Inputs
    ------
    deblended_outputs : HDF5 file from RunDeblender
    eval_blends_real, eval_blends_injected : BlendDataset (truth and metadata)

    Outputs
    -------
    metrics       : yaml with the binned numbers and global summaries
    qa_evaluation : residual-against-variable plots, chi2 distributions, best / median / worst montages

    STUB: nothing is implemented yet. See the guide, sections 14.2 and 14.3.
    """

    name = "Evaluate"
    inputs = [("deblended_outputs", HDFFile), ("eval_blends_real", BlendDataset),
              ("eval_blends_injected", BlendDataset)]
    outputs = [("metrics", YamlFile), ("qa_evaluation", Directory)]
    config_options = {
        "chi2_flag": 1.5,           # chi2 per pixel above which a model is flagged as a failure
    }

    def run(self):
        deblended = self.get_input("deblended_outputs")
        eval_real = self.get_input("eval_blends_real")
        eval_injected = self.get_input("eval_blends_injected")
        metrics_out = self.get_output("metrics")
        qa_out = self.get_output("qa_evaluation")

        # --- planned (dbv_pipeline.metrics: to write)
        # for each evaluation set:
        #     residuals = flux_colour_shape_residuals(deblended[set], truth=eval_*[set])
        #     baseline = flux_colour_shape_residuals(blend_itself, truth=eval_*[set])
        #     binned = bin_residuals(residuals, by=["n_gal", "dmag", "theta", "blendedness", "snr", "psf", "offset"])
        # write_yaml(metrics_out, binned, flag_fraction(chi2_pix, self.config["chi2_flag"]))
        # make_evaluation_qa(residuals, baseline, qa_out)
        raise NotImplementedError("Evaluate is a stub (guide: sections 14.2, 14.3)")
