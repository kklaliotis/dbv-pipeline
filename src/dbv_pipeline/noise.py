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
    noise_qa    : histograms of whitened pixels against a unit Gaussian; chi2 per pixel before and after f_b

    """

    name = "MeasureNoise"
    inputs = [("empty_cutouts", DebvaderDataset)]
    outputs = [("noise_model", YamlFile), ("noise_qa", Directory)]
    config_options = {
        "bands": [str],             
        "estimator": "mad",         # "mad" (robust, default) or "mean"
        "n_stamps": int,
        "likelihood": "measured_noise"  # "measured_noise" (default) or "variance"
    }

    def run(self):
        from .utils.misc import write_yaml

        empty_dir = self.get_input("empty_cutouts")

        # compute the whitened pixels
        whitened = self.whiten_empty_stamps(empty_dir)         # image / sqrt(variance), masked pixels removed
        f_mean, f_mad = self.noise_factors(whitened)                            # per band
        f_b = f_mean if self.config["estimator"] == "mean" else f_mad

        correlations = self.compute_lag1_correlation(whitened)

        info = {
            "bands": self.cfg["bands"],
            "noise_factor": f_b,
            "noise_factor_mean": f_mean,
            "noise_factor_mad": f_mad,
            "lag1_corr": correlations,
            "whitened_mean": np.nanmean(whitened, axis=(0, 1, 2)).tolist(),
            "n_stamps": self.cfg["n_stamps"],
            "likelihood": self.cfg["likelihood"]
        }

        noise_model_out = self.get_output("noise_model")
        noise_qa_out = self.get_output("noise_qa")

        write_yaml(info, noise_model_out.path)
        self.plot_noise_qa(whitened, f_b)  # make QA plots for the noise measurement

        print(f"Noise model written to {noise_model_out.path} and QA plots to {noise_qa_out.path}")

        

    def whiten_empty_stamps(self, empty_dir):
        """
        Read the empty-sky cutouts and return the whitened pixels (image / sqrt(variance)) with masked 
        pixels removed.

        Parameters
        ----------
        empty_dir : DebvaderDataset
            The DebvaderDataset of empty-sky cutouts, with images.npy, variances.npy, and mask.npy.

        Returns
        -------
        whitened : np.ndarray
            The whitened pixels, shape (N, H, W, C), with masked pixels set to NaN.
        """
        # read the images.npy from the empty_dir DebvaderDataset
        images_path = empty_dir.path / "images.npy"
        images = np.load(images_path)  # shape (N, H, W, C) 

        # read the variances.npy from the empty_dir DebvaderDataset
        variances_path = empty_dir.path / "variances.npy"
        variances = np.load(variances_path)  # shape (N, H, W, C)

        # read the mask.npy from the empty_dir DebvaderDataset
        mask_path = empty_dir.path / "mask.npy"
        mask = np.load(mask_path)  # shape (N, H, W, C), bool, True = ignore this pixel

        return np.where(mask, np.nan, images / np.sqrt(variances))  # whitened; masked pixels set to NaN

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
        Compute the lag-1 pixel correlation for each band from the whitened pixels.

        Parameters
        ----------
        whitened : np.ndarray
            The whitened pixels, shape (N, H, W, C), with masked pixels set to NaN.

        Returns
        -------
        correlations : np.ndarray
            The lag-1 pixel correlation per band.
        """
        # flatten the whitened pixels to (N_pixels, N_bands)
        flat = whitened.reshape(-1, whitened.shape[-1])  # shape (N_pixels, N_bands)

        correlations = []
        for b in range(flat.shape[1]):
            band_pixels = flat[:, b]
            valid = ~np.isnan(band_pixels)
            if np.sum(valid) < 2:
                correlations.append(np.nan)
                continue
            x = band_pixels[valid]
            corr = np.corrcoef(x[:-1], x[1:])[0, 1]  # lag-1 correlation
            correlations.append(corr)
        return np.array(correlations)  #shape (N_bands,)

    def plot_noise_qa(self, whitened, f_b):
        """
        Make QA plots for the noise measurement.

        Parameters
        ----------
        whitened : np.ndarray
            The whitened pixels, shape (N, H, W, C), with masked pixels set to NaN.
        f_b : np.ndarray
            The noise factors per band.

        Returns
        -------
        None
        """
        from .qa import make_noise_qa, save

        fig_white, fig_chi2, fig_noise_factor = make_noise_qa(whitened, f_b)

        save(fig_white, self.get_output("noise_qa").path, "whitened_histograms")
        save(fig_chi2, self.get_output("noise_qa").path, "chi2_per_pixel")
        save(fig_noise_factor, self.get_output("noise_qa").path, "noise_factors")

        return None

