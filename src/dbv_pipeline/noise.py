from .base_stage import PipelineStage
from .data_types import DebvaderDataset, Directory, YamlFile

# Heavy or environment-specific imports (numpy, scipy, matplotlib) go inside run().


class MeasureNoise(PipelineStage):
    """
    Measure the noise factor f_b of the likelihood on empty-sky stamps.

    For pure noise the whitened pixels (image / sqrt(variance)) should have unit variance; f_b is the measured
    variance of those pixels per band, so that dividing the variance by f_b makes empty-sky chi2 per pixel
    equal to 1. Also records the lag-1 pixel correlation and the mean of the whitened pixels.

    Inputs
    ------
    empty_cutouts : DebvaderDataset of empty-sky stamps

    Outputs
    -------
    noise_model : yaml with noise_factor per band (and the plain and robust estimators, lag-1 correlation)
    noise_qa    : histograms of whitened pixels against a unit Gaussian; chi2 per pixel before and after f_b

    STUB: nothing is implemented yet. See the guide, section 10.2.
    """

    name = "MeasureNoise"
    inputs = [("empty_cutouts", DebvaderDataset)]
    outputs = [("noise_model", YamlFile), ("noise_qa", Directory)]
    config_options = {
        "bands": [str],             # global
        "estimator": "mad",         # "mad" (robust, default) or "mean"
    }

    def run(self):
        empty_dir = self.get_input("empty_cutouts")
        noise_model_out = self.get_output("noise_model")
        noise_qa_out = self.get_output("noise_qa")

        # --- planned
        # whitened = whiten_empty_stamps(empty_dir)                          # image / sqrt(variance), masked pixels removed
        # f_mean, f_mad = noise_factors(whitened)                            # per band
        # correlations = lag1_correlation(whitened)
        # write_yaml(noise_model_out, choose_estimator(f_mean, f_mad, self.config["estimator"]), correlations)
        # make_noise_qa(whitened, f_mean, f_mad, noise_qa_out)
        raise NotImplementedError("MeasureNoise is a stub (guide: section 10.2)")
