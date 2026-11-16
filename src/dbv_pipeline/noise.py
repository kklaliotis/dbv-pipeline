from .base_stage import PipelineStage
from .data_types import DebvaderDataset, Directory, YamlFile
import numpy as np

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
    qa_noise    : histograms of whitened pixels against a unit Gaussian; chi2 per pixel before and after f_b

    """

    name = "MeasureNoise"
    inputs = [("empty_cutouts", DebvaderDataset)]
    outputs = [("noise_model", YamlFile), ("qa_noise", Directory)]
    config_options = {
        "bands": [str],
        "estimator": "mad",         # "mad" (robust, default) or "mean"
        "likelihood": "measured_noise"  # "measured_noise" (default) or "variance"
    }

    def run(self):
        from pathlib import Path
        from .utils.misc import write_yaml

        if self.config["estimator"] not in ("mad", "mean"):
            raise ValueError(f"estimator must be 'mad' or 'mean', got {self.config['estimator']!r}")

        # ceci gives paths as strings; the Directory output must be created (open_output does that).
        empty_dir = Path(self.get_input("empty_cutouts"))
        noise_model_out = self.get_output("noise_model")
        qa_noise_out = Path(self.open_output("qa_noise", wrapper=True).path)

        # compute the whitened pixels
        whitened = self.whiten_empty_stamps(empty_dir)         # image / sqrt(variance), masked pixels -> NaN
        if whitened.shape[-1] != len(self.config["bands"]):
            raise ValueError(f"{empty_dir} has {whitened.shape[-1]} bands, the configuration {self.config['bands']}")
        f_mean, f_mad = self.compute_noise_factors(whitened)                    # per band
        f_b = f_mean if self.config["estimator"] == "mean" else f_mad

        lag1_x, lag1_y = self.compute_lag1_correlation(whitened)

        info = {
            "bands": list(self.config["bands"]),
            "noise_factor": f_b,
            "estimator": self.config["estimator"],
            "noise_factor_mean": f_mean,
            "noise_factor_mad": f_mad,
            "lag1_corr_x": lag1_x,
            "lag1_corr_y": lag1_y,
            "whitened_mean": np.nanmean(whitened, axis=(0, 1, 2)),
            "n_stamps": int(whitened.shape[0]),
            "n_good_pixels": np.isfinite(whitened).sum(axis=(0, 1, 2)),
            "likelihood": self.config["likelihood"]
        }

        write_yaml(noise_model_out, info)
        self.plot_noise_qa(whitened, f_b, f_mean, f_mad, qa_noise_out)  # make QA plots for the noise measurement

        print(f"Noise model written to {noise_model_out} and QA plots to {qa_noise_out}")



    def whiten_empty_stamps(self, empty_dir):
        """
        Read the empty-sky cutouts and return the whitened pixels (image / sqrt(variance)) with masked
        pixels removed.

        Parameters
        ----------
        empty_dir : pathlib.Path
            The DebvaderDataset directory of empty-sky cutouts, with images.npy, variance.npy, and mask.npy.

        Returns
        -------
        whitened : np.ndarray
            The whitened pixels, shape (N, H, W, C), with masked pixels set to NaN.
        """
        # read the images.npy from the empty_dir DebvaderDataset
        images = np.load(empty_dir / "images.npy")  # shape (N, H, W, C)

        # read the variance.npy from the empty_dir DebvaderDataset
        variances = np.load(empty_dir / "variance.npy")  # shape (N, H, W, C)

        # read the mask.npy from the empty_dir DebvaderDataset
        mask = np.load(empty_dir / "mask.npy")  # shape (N, H, W, C), bool, True = ignore this pixel

        # The mask already covers non-finite and non-positive variance; the second test only keeps sqrt quiet.
        bad = mask | ~(np.isfinite(variances) & (variances > 0)) | ~np.isfinite(images)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(bad, np.nan, images / np.sqrt(variances))  # whitened; masked pixels set to NaN

    def compute_noise_factors(self, whitened):
        """
        Compute the noise factors f_b for each band from the whitened pixels.
        Returns both the mean and MAD estimators.

        Parameters
        ----------
        whitened : np.ndarray
            The whitened pixels, shape (N, H, W, C), with masked pixels set to NaN.

        Returns
        -------
        f_mean : np.ndarray
            The mean of the squared whitened pixels per band (plain estimator).
        f_mad : np.ndarray
            The MAD estimator of the variance of the whitened pixels per band (robust estimator).
        """

        flat = whitened.reshape(-1, whitened.shape[-1])  # shape (N_pixels, N_bands)

        f_mean = np.nanmean(flat**2, axis=0)  # mean of squared whitened pixels per band

        med = np.nanmedian(flat, axis=0)
        f_mad = (np.nanmedian(np.abs(flat - med), axis=0) * 1.4826)**2  # MAD estimator of variance

        return f_mean, f_mad  # each shape (N_bands,)

    def compute_lag1_correlation(self, whitened):
        """
        Compute the lag-1 pixel correlation for each band from the whitened pixels, along rows (x) and along
        columns (y). Only pairs of neighbouring pixels in the same stamp where both pixels are valid are used.

        Parameters
        ----------
        whitened : np.ndarray
            The whitened pixels, shape (N, H, W, C), with masked pixels set to NaN.

        Returns
        -------
        corr_x, corr_y : np.ndarray
            The lag-1 pixel correlation per band, shape (N_bands,) each.
        """
        def corr(a, b):
            out = []
            for c in range(a.shape[-1]):
                x, y = a[..., c].ravel(), b[..., c].ravel()
                ok = np.isfinite(x) & np.isfinite(y)
                out.append(np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 2 else np.nan)
            return np.array(out)

        corr_x = corr(whitened[:, :, :-1, :], whitened[:, :, 1:, :])   # neighbours along a row
        corr_y = corr(whitened[:, :-1, :, :], whitened[:, 1:, :, :])   # neighbours along a column
        return corr_x, corr_y

    def plot_noise_qa(self, whitened, f_b, f_mean, f_mad, qa_dir):
        """
        Make QA plots for the noise measurement.

        Parameters
        ----------
        whitened : np.ndarray
            The whitened pixels, shape (N, H, W, C), with masked pixels set to NaN.
        f_b : np.ndarray
            The noise factors per band (the ones used downstream).
        f_mean, f_mad : np.ndarray
            The two estimators, per band.
        qa_dir : pathlib.Path
            The QA output directory.

        Returns
        -------
        None
        """
        from .qa import make_noise_qa, save

        fig_white, fig_chi2, fig_noise_factor = make_noise_qa(whitened, f_b, f_mean, f_mad,
                                                              list(self.config["bands"]))

        save(fig_white, qa_dir, "whitened_histograms")
        save(fig_chi2, qa_dir, "chi2_per_pixel")
        save(fig_noise_factor, qa_dir, "noise_factors")

        return None
