from .base_stage import PipelineStage
from .data_types import BlendDataset, DebvaderDataset, Directory, ModelDirectory, YamlFile

# Heavy or environment-specific imports (keras, debvader, numpy) go inside run().
# Training keeps its checkpoints in <work_dir>/training/<run_name>, which is NOT a declared output, so that an
# interrupted job can resume (Directory outputs are deleted and recreated by ceci).


class TrainPrior(PipelineStage):
    """
    Train the prior VAE on isolated real galaxies (stage 1 of debvader).

    Encoder and decoder are trained together from scratch with the measured-noise likelihood and the
    forward-modelled PSF: the decoder outputs the galaxy at the internal PSF G_int, and the matching kernel K
    brings it to the observed PSF inside the loss (docs/IMPLEMENTATION_GUIDE.md, section 3).

    Inputs
    ------
    isolated_cutouts : DebvaderDataset
    noise_model      : yaml from MeasureNoise (the noise factor per band)
    kernel_report    : yaml from CheckKernels (also forces the order: no training if the kernel check failed)

    Outputs
    -------
    prior_model : ModelDirectory (weights.h5, config.json, model_card.json)
    qa_prior    : training curves, residual montages, chi2 per pixel against magnitude

    STUB: nothing is implemented yet. See the guide, Phase 5 (section 11) and Phase 6 (section 12).
    """

    name = "TrainPrior"
    inputs = [("isolated_cutouts", DebvaderDataset), ("noise_model", YamlFile), ("kernel_report", YamlFile)]
    outputs = [("prior_model", ModelDirectory), ("qa_prior", Directory)]
    config_options = {
        "run_name": "prior",
        "resume_training": True,
        "work_dir": str,            # global
        "bands": [str],             # global
        "pixel_scale": float,       # global
        "stamp_arcsec": float,      # global
        "internal_psf": dict,       # global
        "kernel_reg": float,        # global
        "debvader": dict,           # data / model / training blocks, translated by utils/config.py
    }

    def run(self):
        isolated_dir = self.get_input("isolated_cutouts")
        noise_model = self.get_input("noise_model")
        kernel_report = self.get_input("kernel_report")
        prior_model_out = self.get_output("prior_model")
        qa_out = self.get_output("qa_prior")

        # --- planned
        # check_consistency(isolated_dir, noise_model, kernel_report, self.config)   # bands, pixel scale, G_int agree
        # debvader_config = build_debvader_config(self.config, isolated_dir, noise_factor, stage=1)   # utils/config.py
        # best_weights = debvader.api.train(debvader_config, stage=1, experiments_dir=work_dir / "training",
        #                                   run_name=self.config["run_name"], resume=self.config["resume_training"])
        # write_model_directory(prior_model_out, best_weights, debvader_config, model_card)
        # make_training_qa(work_dir / "training" / run_name, isolated_dir, prior_model_out, qa_out)
        raise NotImplementedError("TrainPrior is a stub (guide: sections 11 and 12)")


class TrainDeblender(PipelineStage):
    """
    Train the deblender (stage 2 of debvader): a new encoder, initialised from the prior's encoder, maps blended
    scenes to the latent of their central galaxy. The decoder is frozen.

    Same likelihood and forward model as TrainPrior; only the data change: input is the blended scene, target
    is the central galaxy, and the pixel weights come from the blend's own variance plane.

    Inputs
    ------
    blend_cutouts : BlendDataset (training blends)
    noise_model   : yaml from MeasureNoise
    prior_model   : ModelDirectory from TrainPrior (decoder weights, encoder initialisation)

    Outputs
    -------
    deblender_model : ModelDirectory
    qa_deblender    : training curves, blend / target / model / residual montages

    STUB: nothing is implemented yet. See the guide, section 13.3.
    """

    name = "TrainDeblender"
    inputs = [("blend_cutouts", BlendDataset), ("noise_model", YamlFile), ("prior_model", ModelDirectory)]
    outputs = [("deblender_model", ModelDirectory), ("qa_deblender", Directory)]
    config_options = {
        "run_name": "deblender",
        "resume_training": True,
        "work_dir": str,            # global
        "bands": [str],             # global
        "pixel_scale": float,       # global
        "stamp_arcsec": float,      # global
        "internal_psf": dict,       # global
        "kernel_reg": float,        # global
        "debvader": dict,
    }

    def run(self):
        blend_dir = self.get_input("blend_cutouts")
        noise_model = self.get_input("noise_model")
        prior_dir = self.get_input("prior_model")
        deblender_model_out = self.get_output("deblender_model")
        qa_out = self.get_output("qa_deblender")

        # --- planned
        # check_same_frame(prior_dir, blend_dir, self.config)    # bands, stamp size, G_int, scale_per_band must agree
        # debvader_config = build_debvader_config(self.config, blend_dir, noise_factor, stage=2)
        # best_weights = debvader.api.train(debvader_config, stage=2, experiments_dir=work_dir / "training",
        #                                   run_name=self.config["run_name"], resume=self.config["resume_training"],
        #                                   init_weights=prior_dir / "weights.h5")
        # write_model_directory(deblender_model_out, best_weights, debvader_config, model_card)
        # make_training_qa(...)
        raise NotImplementedError("TrainDeblender is a stub (guide: section 13.3)")
