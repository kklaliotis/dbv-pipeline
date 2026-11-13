from .base_stage import PipelineStage
from .data_types import BlendDataset, HDFFile, ModelDirectory, YamlFile

# Heavy or environment-specific imports (keras, debvader, h5py, numpy) go inside run().


class RunDeblender(PipelineStage):
    """
    Run the trained deblender on the evaluation sets.

    For every scene: the model at the internal PSF (g), the model at the observed PSF (m = K * g), the latent
    mean and sigma, and chi2 per pixel against the target and against the blend (the failure flag).
    Point estimates use the encoder mean, not a latent sample.

    Inputs
    ------
    deblender_model      : ModelDirectory from TrainDeblender
    noise_model          : yaml from MeasureNoise
    eval_blends_real, eval_blends_injected : BlendDataset

    Outputs
    -------
    deblended_outputs : HDF5 file, one group per evaluation set (g, m, z_mean, z_sigma, chi2_pix, ...)

    STUB: nothing is implemented yet. See the guide, section 14.1.
    """

    name = "RunDeblender"
    inputs = [("deblender_model", ModelDirectory), ("noise_model", YamlFile),
              ("eval_blends_real", BlendDataset), ("eval_blends_injected", BlendDataset)]
    outputs = [("deblended_outputs", HDFFile)]
    config_options = {
        "batch_size": 512,
        "eval_sets": ["eval_blends_real", "eval_blends_injected"],
    }

    def run(self):
        model_dir = self.get_input("deblender_model")
        noise_model = self.get_input("noise_model")
        eval_dirs = {tag: self.get_input(tag) for tag in self.config["eval_sets"]}
        outputs_path = self.get_output("deblended_outputs")

        # --- planned (debvader.inference.predictor.DebvaderPredictor.predict_forward, guide section 11.8)
        # predictor = load_predictor(model_dir)          # config.json + weights.h5 + model_card.json
        # with self.open_output("deblended_outputs") as h5:
        #     for tag, directory in eval_dirs.items():
        #         result = predictor.predict_forward(images, psf, variance, mask, batch_size=...)
        #         write_group(h5, tag, result)           # g, m, z_mean, z_sigma, chi2_pix, chi2_pix_target
        raise NotImplementedError("RunDeblender is a stub (guide: section 14.1)")
