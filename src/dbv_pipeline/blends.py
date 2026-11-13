from .base_stage import PipelineStage
from .data_types import BlendDataset, DebvaderDataset

# Heavy or environment-specific imports (galsim, numpy, debvader) go inside run().


class BuildBlends(PipelineStage):
    """
    Build blended scenes by injecting synthetic contaminants around real galaxies.

    Three datasets are made, all from the isolated stamps:
      blend_cutouts         training set: the real isolated galaxy is the central object, the target is its
                            untouched (noisy) stamp                                  (central = "real")
      eval_blends_real      the same construction, from the held-out test split       (central = "real")
      eval_blends_injected  evaluation set with exact truth: a synthetic central galaxy injected into real
                            empty sky, the target is its noiseless render             (central = "injected")
    Variance planes are carried along, with the shot noise of the injected light added.

    Inputs
    ------
    isolated_cutouts : DebvaderDataset
    empty_cutouts    : DebvaderDataset (empty-sky stamps for the injected-central set)

    Outputs
    -------
    blend_cutouts, eval_blends_real, eval_blends_injected : BlendDataset

    STUB: nothing is implemented yet. See the guide, sections 13.1 and 13.2.
    """

    name = "BuildBlends"
    inputs = [("isolated_cutouts", DebvaderDataset), ("empty_cutouts", DebvaderDataset)]
    outputs = [("blend_cutouts", BlendDataset), ("eval_blends_real", BlendDataset),
               ("eval_blends_injected", BlendDataset)]
    config_options = {
        "n_train_scenes": 100000,
        "n_eval_scenes": 10000,
        "dmag": [-1.0, 3.0],        # contaminant minus target magnitude, drawn uniformly
        "theta": [0.325, 3.0],      # separation in arcsec, drawn uniformly
        "seed": 0,
    }

    def run(self):
        isolated_dir = self.get_input("isolated_cutouts")
        empty_dir = self.get_input("empty_cutouts")
        train_out = self.get_output("blend_cutouts")
        eval_real_out = self.get_output("eval_blends_real")
        eval_injected_out = self.get_output("eval_blends_injected")

        # --- planned (debvader.data.blends.build_blends, from scripts/real_data/build_blends.py: to refactor)
        # build_blends(isolated_dir, train_out, n_scenes=self.config["n_train_scenes"],
        #              central="real", splits=[0, 1], dmag=..., theta=..., seed=...)
        # build_blends(isolated_dir, eval_real_out, n_scenes=self.config["n_eval_scenes"],
        #              central="real", splits=[2], seed=seed + 1)
        # build_blends(isolated_dir, eval_injected_out, empty_dir=empty_dir, n_scenes=...,
        #              central="injected", splits=[2], seed=seed + 2)
        # blend_qa(train_out, eval_real_out, eval_injected_out)
        raise NotImplementedError("BuildBlends is a stub (guide: sections 13.1, 13.2)")
