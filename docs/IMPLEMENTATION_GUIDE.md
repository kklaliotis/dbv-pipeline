# Rubin-only deblending with debvader (Test I): implementation guide

Written 2026-11-12. Supersedes `debvader_rubin_test1_brief.md` wherever the two disagree.

This guide covers three repositories:

| Repo | Path | Role |
|---|---|---|
| `dbv-pipeline` | `/global/cfs/cdirs/m1727/laliotis/debvader_pipeline/dbv-pipeline` | ceci orchestration only: config translation, calling library functions, declared outputs, QA |
| `DP2-cutouts` | `/global/cfs/cdirs/m1727/laliotis/debvader_pipeline/DP2-cutouts` | selection, isolation, stamps, PSF, variance, masks, export |
| `debvader-keras` | `~/Software/debvader-keras` (branch `lia_tests`) | model, likelihood, forward model, loader, training, blend building, prediction |

Contents

1. Decisions and scope
2. Facts verified against the code and the DP2 Butler
3. Mathematical formalism
4. Architecture overview
5. Environments
6. Phase 0: repair and restructure `dbv-pipeline`
7. Phase 1: ceci skeleton with stub stages
8. Phase 2: DP2-cutouts, Butler source and multi-patch stamps
9. Phase 3: DP2-cutouts, export to the debvader dataset format
10. Phase 4: PSF, noise and kernel validation stages
11. Phase 5: debvader-keras, forward model and measured-noise likelihood
12. Phase 6: `TrainPrior`
13. Phase 7: blends and `TrainDeblender`
14. Phase 8: `RunDeblender` and `Evaluate`
15. Phase 9: scale to the full ECDFS field
16. Master checklist of files
17. Gotchas
18. Open items

---

## 1. Decisions and scope

Test I goal: set up the architecture, train on real Rubin data, validate the method. The weights are not meant to be reused for the joint Rubin+Euclid model.

| Topic | Decision |
|---|---|
| Training data | Real DP2 only, ECDFS / EDF-F field. No simulation stage (the old "Stage Ia" is dropped). |
| Start point | From scratch. No warm start from the BTK-trained weights. |
| Likelihood | Measured noise from the variance plane, no learned sigma. Identical in every stage. |
| PSF | Forward-modelled. Decoder output is the galaxy at a fixed internal PSF `G_int`; a per-object kernel `K` takes it to the observed PSF inside the loss. |
| `G_int` | A **config value**, validated by `MeasurePsf`. Never chosen by a pipeline stage. |
| `K` | Computed in the data loader from the PSF stamp. `CheckKernels` only validates. |
| Normalisation | Encoder input keeps `tanh(arcsinh(.))`. Decoder output is **linear**, in units of a fixed per-band scale `s_b`. |
| Stamp size | Config value in arcsec (11.8 arcsec). Pixel count derived per instrument and **always odd** (59 px for Rubin). |
| Stage II target | Switch: `central: real` (Anna's design, used for training) or `central: injected` (noiseless truth, used for evaluation). |
| Heads | Ellipticity, redshift and PSF-conditioned heads off. |
| Encoder PSF input | Deferred. |
| Unrecognised blends | Not in training. Flagged in evaluation using the Euclid match that DP2-cutouts already computes. |
| Joint model (later) | Option A: one decoder on a common fine model grid with per-instrument observation operators. Test I only has to avoid blocking it. |
| Code location | Real code in `DP2-cutouts` and `debvader-keras` (branches, later PRs). `dbv-pipeline` orchestrates. |
| Pipeline shape | One repo, three ceci pipelines: `data_dp2.yml`, `train.yml`, `eval.yml`. |

What "do not block Option A" means in practice:

- The forward model takes model pixel scale, observed pixel scale and `G_int` as explicit parameters. Test I is the special case where the two scales are equal.
- Datasets carry instrument name, band list and pixel scale in a sidecar file, and nothing downstream hard-codes 59, 0.2 or `ugrizy`.

---

## 2. Facts verified against the code and the DP2 Butler

Everything in this section was checked on 2026-11-12. Re-check anything that looks different when you get there.

### 2.1 DP2 Butler at NERSC

Setup: `bash ~/setup.sh python ...` (LSST stack from CVMFS, `DAF_BUTLER_REPOSITORY_INDEX` set).

| Item | Value |
|---|---|
| Repo alias | `dp2` |
| Collection | `dp2` |
| Skymap | `lsst_cells_v2`, 10 x 10 patches per tract |
| Coadd dataset type | `deep_coadd`, dimensions `{band, skymap, tract, patch}` |
| Coadd Python type | `lsst.images.cells.CellCoadd` (**not** `ExposureF`) |
| Object table | `object`, dimensions `{skymap, tract}`, storage class `ArrowAstropy`, 1224 columns |
| ECDFS test patch | tract 5063, patch 34 (all six bands present) |
| Patch geometry | outer bbox 3300 x 3300 px, inner 3000 x 3000 px, so a 150 px overlap border |
| Units | `nJy` (`coadd.unit`) |

`CellCoadd` API that the cutout code needs:

```python
from lsst.daf.butler import Butler
from lsst.images import Box

butler = Butler("dp2", collections="dp2")
did = dict(band="i", skymap="lsst_cells_v2", tract=5063, patch=34)
coadd = butler.get("deep_coadd", did)

coadd.bbox                      # Box [y=8850:12150, x=11850:15150], tract pixel frame
coadd.image.array               # (3300, 3300) float32, nJy
coadd.variance.array            # (3300, 3300) float32, nJy^2
coadd.mask.get("DETECTED")      # (3300, 3300) bool; mask.array itself is (H, W, 2) uint8
coadd.noise_realizations[0].array   # one noise realisation, same shape as the image
coadd.psf.compute_kernel_image(x=13500.0, y=10500.0)   # tract pixel coords -> Image
coadd.fits_wcs                  # astropy WCS in LOCAL array coordinates (see below)

# Partial read (verified): a 59 x 59 stamp without loading the patch
sub = butler.get("deep_coadd", did, parameters={"bbox": Box.factory[10471:10530, 13471:13530]})
```

Measured on tract 5063, patch 34, band i:

- Mask planes: `NO_DATA, INTERPOLATED, COSMIC_RAY, SATURATED, DETECTION_EDGE, CLIPPED, REJECTED, DETECTED, INEXACT_PSF`.
- PSF kernel image: 35 x 35, float64, sum 1.00001, peak at pixel (17, 17), bbox `[y=-17:18, x=-17:18]`. FWHM about 1.03 arcsec (from the area above half maximum).
- The PSF is a `CellPointSpreadFunction`: one PSF per 150 px cell, so it is piecewise constant over 30 arcsec.
- `fits_wcs` gave (53.285, -27.862) for pixel (0, 0), consistent with a WCS local to the array (index 0 = `bbox.start`). Confirm with a round-trip test in Phase 2.
- Variance calibration on undetected, non-`NO_DATA` pixels:

| Quantity | Value |
|---|---|
| median variance plane | 10.91 nJy^2 |
| robust (MAD) variance of the image | 11.22 nJy^2 |
| mean of `image^2 / variance` | 1.079 |
| mean of `noise_realization^2 / variance` | 1.0005 |
| lag-1 correlation of the noise realisation | 0.15 |

So the variance plane is correct for pure noise. The 8 % excess on the real image is consistent with light from undetected sources.

Other dataset types that exist and may be useful later:

- `deep_coadd_cell_systematics_noisecorr_6bands` (per patch) and `deep_coadd_systematics_noisecorr_6bands_tractavg` (per tract): noise correlation products, candidates for the L2 likelihood.
- `object_scarlet_models`: Rubin scarlet children, the baseline deblender.
- `injected_deep_coadd`: the type is registered but has **no datasets** in tract 5063. We do our own stamp-level injection.

Columns confirmed present in `object`: `objectId, parentObjectId, coord_ra, coord_dec, tract, patch, detect_isIsolated, refExtendedness, i_cModelFlux, i_ixx, i_ixxPSF, i_blendedness, i_extendedness, r_hsmShapeRegauss_e1`. `detect_isPrimary`, `x`, `y` and `deblend_nChild` were not found under those names. Read the full list with `butler.get("object.columns", dataId)` before writing the column list.

### 2.2 debvader-keras (branch `lia_tests`)

| File | What is there now |
|---|---|
| `src/debvader/core/decoder.py` | Final layer `Conv2D(filters=2*C, kernel_size=3, activation="relu")`, then an asymmetric crop 64 -> 59. Output is `(B, H, W, 2C)`: first half mean, second half scale. Both halves are >= 0 because of the ReLU. The `normalization` argument is unused. The first dense layer is hard-coded to 560 units. |
| `src/debvader/core/encoder.py` | `BatchNormalization`, four conv blocks (stride 2), flatten, PReLU, dense to `D + D(D+1)/2`. |
| `src/debvader/modules/layers.py` | `KerasMultivariateNormalTriL`: builds a Cholesky factor, **always samples** `z = mu + L eps` (there is no `training` switch), adds `kl_weight * mean(KL)` through `add_loss`. `KerasIndependentNormal.compute_log_prob`: `scale = 1e-4 + raw_scale`; `scale_max` is stored and never used. |
| `src/debvader/core/vae.py` | `call()` returns the decoder tensor (or a dict when heads are on). `_compute_recon_loss` returns `-mean(log_prob) / N_pix`. `compute_loss` = `recon * w_recon + sum(self.losses) + heads`. |
| `src/debvader/data/normalization.py` | `NonLinearNorm`: `tanh(arcsinh(2.5 * x / s_b))`, `s_b` from `stats.json` (`nonlinear.scale_per_band`, the 99.9th percentile). |
| `src/debvader/data/loader.py` | `DebvaderDataLoader` recognises a "real build" directory (`images.npy` + `meta.npz`, optional `targets.npy`). `_fetch_real` selects by `meta["split"]` (0 train, 1 val, 2 test). `DebvaderBatchDataset` (a `keras.utils.PyDataset`) yields `(x, y)` with optional labels and applies `augment_data`. Inputs **and targets** are normalised on CPU in `load_data`. |
| `src/debvader/data/augmentation.py` | Random left-right flip, up-down flip and `rot90`, applied identically to `x` and `y`. |
| `src/debvader/training/trainer.py` | `DebvaderTrainer.__init__` builds the normaliser, loader, model and optimiser. `train_stage1` reads splits `train_clean` / `val_clean`. `train_stage2` loads `stage1_best.weights.h5` from the same run directory, sets `model.decoder.trainable = False`, recompiles, reads `train_blended` / `val_blended`. Both pass `steps_per_epoch = train_size // batch_size`. |
| `src/debvader/utils/paths.py` | `DATA_DIR`, `EXPERIMENTS_DIR` are read from the environment **at import time**. Run outputs go to `EXPERIMENTS_DIR/<run_name>/{checkpoints,logs,outputs}`. |
| `src/debvader/utils/config.py` | Dataclasses `DataConfig`, `ModelConfig`, `TrainingConfig`. `Config.from_json` **silently drops unknown keys**. |
| `scripts/real_data/build_blends.py` | Reads a real build (`images.npy`, `psf.npy`, `meta.npz`), injects 0 to 3 GalSim contaminants per scene with shot noise from `Var = a + b * Flux`, writes `images.npy` (blend), `targets.npy` (untouched stamp), `psf.npy`, `meta.npz`. Needs `metadata_magnitude`, `metadata_e1`, `metadata_e2`, `split`; uses `hsm_sigma`, `hsm_resolution` when present. |
| `scripts/recompute_stats.py` | Writes `stats.json` for a dataset directory. |
| Places that assume `2C` output channels | `vae.py:312`, `inference/predictor.py:183` and `:217`, `training/callbacks.py:575`, `:1450`, `:1501`. |

Other observations:

- The comment in `vae.py` ("mvn returns mu when training=False") is out of date: the layer always samples. For inference, take `mu` from the encoder output directly.
- On the login node the `debvader` conda env reports "a CUDA-enabled jaxlib is not installed. Falling back to cpu". Fix this before any GPU training (section 5).
- `import keras` fails in that env unless `KERAS_BACKEND=jax` is set.

### 2.3 DP2-cutouts

| File | What is there now |
|---|---|
| `src/dp2_cutouts/cli.py` | `dp2-cutouts <config> <step>`; steps `rubin_catalog, euclid_catalog, copaings, rubin_images, catalog_table, stamps, package`. |
| `src/dp2_cutouts/config.py` | `Config.load(path)`; `cfg["section"]`; `cfg.data(*parts)` returns a path under `data_root` and creates the parent. |
| `sources/rubin_tap.py` | Object catalogue for a sky box through TAP (needs an RSP token). Writes `rubin_data/catalog/object_box.parquet`. |
| `sources/rubin_soda.py` | One SODA cutout per band of **one region of one patch**. Writes six FITS files and `images.json`. |
| `sources/euclid_hf.py` | Euclid Q1 MER catalogue and photo-z from HuggingFace. |
| `stages/run_copaings.py` | copaings matching and blending entropy. |
| `stages/catalog_table.py` | One row per Rubin object; `cut1`, `cut2`, `cut6` values and pass flags; PSF **moments** per band (no PSF images). |
| `stages/stamps.py` | `load_images(cfg)` returns a dict (`flux, var, mask, detected, wcs, x0, y0, bits, meta`). `process_target` applies cuts 3, 4, 5, 7, 8, masks companions and fills them with real empty sky (`SkyBank`), and returns `flux_raw, flux, ivar, maskbits, companion_mask`. `run(cfg)` handles a single (tract, patch). Only targets passing `cut1` and `cut2` are processed. |
| `stages/package.py` | MultimodalUniverse parquet, partitioned by HEALPix. |

Hard-coded values to remove: `0.2` (pixel scale) in `rubin_soda.py`, `stamps.py`, `package.py`; `stamp_px: 59`, `margin_px: 29`, `diag_px: 117` in the YAML.

### 2.4 dbv-pipeline

- `pyproject.toml` requires `setuptools_dbv`, which does not exist. It must be `setuptools_scm` (a third-party package that derives the version from git; unrelated to the "SCM" pipeline name).
- `src/dbv_pipeline/__init__.py` imports `_version`, which is only generated by `setuptools_scm`.
- `example/ceci_dummy_stage.yml` has `modules: scm_pipeline`.
- `main.yaml` lists `modules: [select, cutouts, ...]`; ceci needs importable names.
- `utils/provenance.py` imports `distutils.version`, which was removed in Python 3.12. The LSST stack runs a newer Python, so this import will fail there.
- `data_types/base.py`: `Directory.read_provenance` calls `yaml.load(f)` without a loader (an error in PyYAML >= 6) and catches `KeyError` instead of `FileNotFoundError`; `FileCollection.read_listing` opens the file in write mode; `Directory.open(path, "w")` deletes an existing directory.
- `select.py`, `cutouts.py`, `psf.py` call undefined placeholder functions.

### 2.5 ceci (version 2.6, as installed in your TXPipe env)

- A stage is found by importing the modules listed under `modules:`. One module may define any number of stages.
- A stage is launched as `python3 -m <top-level package> <StageName> --<tag>=<path> ... --config=<file>`. That is why `dbv_pipeline/__main__.py` must exist.
- Config: the `global` section and the stage's own section are merged flat into `self.config`. Keys not declared in `config_options` are **passed through unchanged**, including nested dictionaries.
- Outputs are written under an `inprogress_` name and renamed when `run()` returns without error. With `resume: True`, a stage whose outputs all exist is skipped; this does not cascade.
- Stage entries accept `name`, `classname`, `module_name`, `aliases`, `nprocess`, `threads_per_process`, `nodes`.
- CLI: `ceci <pipeline.yml>`, `ceci --dry-run <pipeline.yml>`, `ceci --flow-chart <file.png> <pipeline.yml>`.

---

## 3. Mathematical formalism

### 3.1 Generative model

For one object, with bands `b = 1..C` and pixels `i`:

```
z      ~ N(0, I_D)                       latent, D = 32
d(z)   = Decoder_theta(z)                dimensionless, >= 0, shape (H, W, C)
g_b    = s_b * d_b(z)                    galaxy at the internal PSF G_b, in nJy
m_b    = A_b[g_b]                        model in the observed frame
x_b,i  = m_b,i + n_b,i                   data
n_b,i  ~ N(0, f_b * sigma_b,i^2)         independent per pixel (L1 approximation)
```

- `s_b` is a fixed per-band flux scale (section 3.5).
- `sigma_b,i^2` is the variance plane; `f_b` is a per-band factor close to 1 (section 3.6).
- `A_b` is the observation operator. In Test I the model grid equals the data grid, and

```
A_b[g] = K_b * g          (2D convolution)
K~_b(k) = P~_b(k) / G~_b(k)
```

  where `P_b` is the object's PSF stamp and `G_b` the internal PSF. Then `K_b * G_b = P_b`, so a point source rendered by the decoder at `G_b` appears with PSF `P_b`.
- For the joint model, `A_b` becomes "convolve with `K_b` on the model grid, then resample to the instrument's pixels". Keep the name and the interface; only the body changes.

### 3.2 Likelihood and the training loss

Negative log-likelihood, dropping terms that do not depend on the model:

```
-log p(x | z) = (1/2) * chi2(x, z) + const

chi2 = sum_b sum_i  w_b,i * ( x_b,i - (K_b * g_b)_i )^2

w_b,i = M_b,i / ( f_b * sigma_b,i^2 )         M = 1 on good pixels, 0 on masked ones
```

The VAE objective (negative ELBO) for one object:

```
L = E_{q_phi(z|x)} [ (1/2) chi2(x, z) ]  +  beta * KL( q_phi(z|x) || N(0, I) )
```

with `beta = 1` for the exact ELBO. The encoder posterior is a full-covariance Gaussian `q = N(mu, L L^T)`, for which

```
KL = (1/2) * ( tr(L L^T) + mu^T mu - D - 2 * sum_j log L_jj )
```

This is what `KerasMultivariateNormalTriL` already computes.

**What the code minimises.** The existing code divides the reconstruction term by `N_pix = H * W * C`:

```
loss_code = w_recon * (1/2) chi2 / N_pix  +  kl_weight * KL
```

So `kl_weight = beta / N_pix`. For 59 x 59 x 6, `N_pix = 20886`, and the exact ELBO is

```
kl_weight = 1 / 20886 = 4.79e-5
```

Anna's real-data config uses `1e-4`, about twice the ELBO value. Start at `4.79e-5` and treat it as the reference point when tuning.

Keep the division by the constant `N_pix`, not by the number of unmasked pixels, so that `kl_weight` means the same thing for every stamp.

**Working in normalised units.** Divide everything by `s_b`:

```
x'_b = x_b / s_b
w'_b,i = s_b^2 * M_b,i / ( f_b * sigma_b,i^2 )
chi2 = sum_b sum_i  w'_b,i * ( x'_b,i - (K_b * d_b)_i )^2
```

The data loader supplies `x'`, `w'` and `K`; the loss never sees `s_b`, `f_b` or `sigma` separately. This removes the "variance in the wrong units" failure mode: the three quantities are built in one function.

**Diagnostic.** `chi2 / N_good`, with `N_good = sum M`, should approach 1 on validation data when the model is adequate. A value well above 1 means unmodelled structure; well below 1 means the model is fitting noise.

### 3.3 Why noisy targets are the right thing in Stage I

In Stage I the target is the same noisy image as the encoder input. The objective above is then the standard ELBO of a generative model whose likelihood contains the true noise. The decoder learns the distribution of noise-free, `G_int`-resolution galaxies because the noise is accounted for in `p(x|z)`, not because of any property of the encoder.

Two consequences:

- The likelihood must be identical in every stage, since it defines what the decoder output means.
- The brief's argument that "the cross term vanishes" only holds when the target noise is independent of the encoder input. It does not hold in Stage I; the ELBO argument replaces it.

### 3.4 The internal PSF and the kernel

Requirements on `G_b`:

1. Narrower than every observed PSF in band `b`, in training and science data.
2. Well sampled on the model grid: FWHM >= 2.5 px (0.5 arcsec at 0.2 arcsec/px).
3. As close to the narrowest real PSF as 1 and 2 allow. The further below the data resolution it sits, the more the decoder output is determined by the prior.
4. Fixed for all stages and all datasets.
5. `|K~(k)| <= 1 + eps` for every object and band, checked numerically.

**Choice of profile.** `K~ = P~ / G~` must not exceed 1 at any frequency.

- A Gaussian `G` falls as `exp(-k^2)`, faster than an atmospheric PSF at high `k`, so the ratio can exceed 1 at high `k`.
- A Moffat with a small index (heavy wings, e.g. `beta_M = 2.5`) has a larger second moment than a real PSF of similar FWHM, so `G~` drops faster near `k = 0` and the ratio can exceed 1 at **low** `k`. For a Moffat, `<r^2> = alpha^2 / (beta_M - 2)`.
- A Moffat with `beta_M` close to that of the atmosphere (about 4.765 for Kolmogorov turbulence) has a functional form similar to the numerator and behaves best.

These are possible failure modes, not observed ones. In a quick check on 35 px stamps (Moffat `beta_M = 3.5` PSFs of FWHM 0.6 to 1.0 arcsec against a 0.5 arcsec `G` that was Gaussian, Moffat 2.5 or Moffat 4.765), `max |K~|` stayed at 1.0 in every case. Real coadd PSFs are less regular, so `CheckKernels` on the real stamps is what decides.

Start with a circular Moffat, `beta_M = 4.765`, FWHM 0.5 arcsec in all bands. Section 2.1 found an i-band PSF of about 1.0 arcsec in one patch, so there is margin; `MeasurePsf` decides whether the FWHM can be raised.

Moffat profile, sampled at pixel centres and normalised to unit sum:

```
G(r) = ( 1 + r^2 / alpha^2 )^(-beta_M)
alpha = FWHM / ( 2 * sqrt( 2^(1/beta_M) - 1 ) )
```

Both `P` (from the coadd) and `G` (sampled at pixel centres) are *effective* PSFs: the pixel response is already included. That is correct as long as the model grid equals the data grid.

**Regularised kernel.** Division by `G~` is unstable where `G~` is tiny, so use a Tikhonov form:

```
K~(k) = P~(k) * conj(G~(k)) / ( |G~(k)|^2 + lambda^2 ),      lambda = 1e-3
```

Where `|G~| >> lambda` this equals `P~/G~`. Where `|G~| << lambda` it goes smoothly to zero. Requirement 1 guarantees `P~` is already small there.

Two checks per object and band:

```
max_k |K~(k)| <= 1 + eps_K                      eps_K = 0.02
max_i | (K * G)_i - P_i | / max_i P_i <= eps_R  eps_R = 1e-3
```

**What is trustworthy in the output.** Structure in `g` at or coarser than the observed PSF is constrained by data. Structure finer than the observed PSF comes from the prior. Reconvolving `g` to the observed PSF or any wider one is safe; rendering it narrower than observed is extrapolation.

### 3.5 Normalisation

Two separate transforms, sharing one set of constants `s_b`:

```
encoder input:   u_b = tanh( arcsinh( 2.5 * x_b / s_b ) )      (existing NonLinearNorm)
decoder output:  g_b = s_b * d_b                                (linear)
```

- The encoder input transform is free: it only changes what features the encoder sees.
- The decoder output must be linear in flux because convolution and chi-squared are linear operations.
- `s_b` is the 99.9th percentile of the pixel values of the isolated training split in band `b`, as `recompute_stats.py` already computes. It is stored in `stats.json` in the dataset directory and copied into the model directory.

### 3.6 Noise factor

On empty-sky stamps `e` (no detected pixel, no bad pixel), with model = 0:

```
f_b = < e_b,i^2 / sigma_b,i^2 >        (plain mean)
f_b (robust) = ( 1.4826 * MAD( e_b,i / sigma_b,i ) )^2
```

After dividing by `f_b`, empty-sky `chi2 / pixel = 1` by construction. Record both estimators; use the robust one.

What section 2.1 implies:

- The variance plane is already right for pure noise (`1.0005`), so `f_b` is not correcting the variance plane.
- The excess (`1.08` in i) is undetected faint sources, which the model cannot represent and which act as extra noise.
- Noise correlation (lag-1 about 0.15) does not change the *mean* of chi-squared per pixel. It changes its scatter and the effective number of independent pixels, so it affects how `beta` should be tuned, not `f_b`.

**L2 hook (not implemented in Test I).** With whitened residual `rho_b = M_b * (x_b - m_b) / sigma_b`:

```
chi2_L2 = sum_b sum_k | FFT(rho_b)_k |^2 / S_b,k
S_b,k = < | FFT( e_b / sigma_b )_k |^2 >  over empty-sky stamps, same size and FFT convention
```

Leave a config key `likelihood: measured_noise | measured_noise_fourier` and raise `NotImplementedError` for the second.

### 3.7 Stage II objectives

The decoder is frozen. A new encoder `phi'`, initialised from the Stage I encoder, is trained on blended scenes `y`:

```
L_II = E_{q_phi'(z|y)} [ (1/2) * sum w' * ( t' - K * d(z) )^2 ] / N_pix  +  kl_weight * KL
```

| Mode | Scene `y` | Target `t` | Weights from |
|---|---|---|---|
| `central: real` | real isolated stamp + injected contaminants (+ their shot noise) | the untouched real stamp (noisy) | variance of the blend = stamp variance + `b_b * contaminant flux` |
| `central: injected` | real empty-sky stamp + injected central + injected contaminants (+ shot noise) | noiseless render of the central at the local PSF | variance of the blend = empty-sky variance + `b_b * all injected flux` |

- `central: real` keeps real morphology, real noise and the real PSF in the target. It is the training set.
- `central: injected` has exact truth (fluxes, shapes, positions) but parametric centrals. It is an evaluation set.
- Shot noise of injected light uses the affine variance law `Var = a + b * Flux` already measured by `measure_variance_law.py`.

### 3.8 Inference outputs

For a stamp `y` with PSF `P`:

```
(mu, L) = Encoder(u(y))
g       = s * Decoder(mu)                    model at G_int
m       = K * g                              model at the observed PSF
sigma_z = sqrt( diag( L L^T ) )              latent uncertainty
chi2_pix = sum w * (y - m)^2 / N_good        failure flag (uses the unnormalised y, m and sigma)
```

Use `mu`, not a sample, for point estimates.

Flux redistribution, when every member `j` of a blend has been run as the central object and the models are placed on a common scene grid:

```
child_i = y * m_i / sum_j m_j
```

This needs scene-level cutouts and is deferred (section 14.4).

---

## 4. Architecture overview

```
                 data_dp2.yml  (LSST stack env)
  DP2 Butler, Euclid Q1
        |
   SelectSources ----> selection_catalog
        |
   MakeCutouts ------> isolated_cutouts, empty_cutouts        (DP2-cutouts)
        |                    |               |
   MeasurePsf -> psf_stats   |          MeasureNoise -> noise_model
                             |
- - - - - - - - - - - - - - -|- - - - - - - - - - - - - - - - - - - - - - - -
                 train.yml  (debvader env, GPU)
   CheckKernels -> kernel_report
   TrainPrior ---> prior_model                                 (debvader-keras)
   BuildBlends --> blend_cutouts, eval_blends_real, eval_blends_injected
   TrainDeblender -> deblender_model
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 eval.yml  (debvader env)
   RunDeblender -> deblended_outputs
   Evaluate -----> metrics, qa_evaluation
```

Outputs of one pipeline are listed under `inputs:` of the next, by path.

### 4.1 The dataset contract

One format is shared by DP2-cutouts (writer), debvader-keras (reader, blend builder) and the pipeline stages. It is debvader's existing "real build" directory, extended:

```
<dataset>/
  images.npy      (N, H, W, C) float32   flux, nJy                         [exists today]
  variance.npy    (N, H, W, C) float32   variance plane, nJy^2             [new]
  mask.npy        (N, H, W, C) bool      True = ignore this pixel          [new]
  psf.npy         (N, P, P, C) float32   PSF stamp, each band sums to 1    [exists today]
  targets.npy     (N, H, W, C) float32   blends only: what to recover      [exists today]
  meta.npz        per-object columns (below)                               [exists, extended]
  stats.json      {"nonlinear": {"scale_per_band": [...]}}                 [exists today]
  dataset.json    instrument, bands, pixel_scale, stamp_arcsec, stamp_px,
                  psf_px, unit, kind, provenance                           [new]
```

`meta.npz` columns:

| Key | Meaning | Needed by |
|---|---|---|
| `split` | 0 train, 1 validation, 2 test; assigned by patch | loader, blend builder |
| `metadata_id` | Rubin `objectId` | everything |
| `ra`, `dec`, `tract`, `patch` | position | QA, evaluation |
| `offset_x`, `offset_y` | sub-pixel offset of the object from the stamp centre | evaluation |
| `metadata_magnitude` | i-band cModel magnitude | blend builder |
| `metadata_e1`, `metadata_e2` | shape of the object on the stamp (HSM) | blend builder |
| `hsm_sigma`, `hsm_resolution` | size and resolution factor | blend builder |
| `masked_fraction` | fraction of pixels filled with empty sky | QA |
| `sb_euclid`, `has_euclid_counterpart`, `cut*_pass`, `cut*_value` | isolation and blend flags | evaluation |
| `psf_fwhm` | (N, C) FWHM per band, arcsec | `MeasurePsf` |

`dataset.json` example:

```json
{
  "instrument": "rubin",
  "bands": ["u", "g", "r", "i", "z", "y"],
  "pixel_scale": 0.2,
  "stamp_arcsec": 11.8,
  "stamp_px": 59,
  "psf_px": 35,
  "unit": "nJy",
  "kind": "isolated",
  "source": {"repo": "dp2", "collection": "dp2", "skymap": "lsst_cells_v2"}
}
```

Conventions that everything depends on:

- Arrays are channel-last, indexed `[row, column, band]`.
- The object sits on the centre pixel `(H // 2, W // 2)` to within half a pixel; `offset_x`, `offset_y` record the remainder.
- The PSF stamp is centred on pixel `(P // 2, P // 2)`.
- `mask` is True for `NO_DATA`, `SATURATED`, `COSMIC_RAY`, `DETECTION_EDGE`, and any pixel with non-finite or non-positive variance. Pixels filled with empty sky (`FILLED_SKY`) are **valid**: they hold real sky with the correct variance.
- For the joint test, a dataset becomes a directory with one such sub-directory per instrument (`rubin/`, `euclid/`), row-aligned.

Stamp size rule (put this function in both libraries; the pipeline passes the result):

```python
def odd_pixels(size_arcsec: float, pixel_scale: float) -> int:
    """Pixel count for an angular stamp size, always odd so that a centre pixel exists."""
    n = int(round(size_arcsec / pixel_scale))
    return n if n % 2 == 1 else n + 1

# odd_pixels(11.8, 0.2) -> 59;  odd_pixels(11.8, 0.1) -> 119
```

---

## 5. Environments

Two environments, each with `ceci` and `dbv-pipeline` installed.

| Environment | Used by | Must contain |
|---|---|---|
| LSST stack (`~/setup.sh`) | `data_dp2.yml` | `lsst_distrib`, `ceci`, `dbv-pipeline`, `dp2-cutouts` and its dependencies (copaings, friendly, lsdb, healpy, galsim) |
| `debvader` conda env | `train.yml`, `eval.yml` | `debvader-keras`, `ceci`, `dbv-pipeline`, `galsim`, `h5py`, GPU-enabled `jaxlib` |

### 5.1 Data environment

The LSST stack on CVMFS is read-only, so install the extra packages into a user prefix layered on top of it. Keep the prefix on CFS, not in `$HOME` (40 GB quota).

```bash
# one-off
export DBV_DATA_PREFIX=/global/cfs/cdirs/m1727/laliotis/conda/lsst-extras
bash ~/setup.sh pip install --prefix $DBV_DATA_PREFIX ceci
bash ~/setup.sh pip install --prefix $DBV_DATA_PREFIX --no-deps -e /global/cfs/cdirs/m1727/laliotis/debvader_pipeline/dbv-pipeline
bash ~/setup.sh pip install --prefix $DBV_DATA_PREFIX -e /global/cfs/cdirs/m1727/laliotis/debvader_pipeline/DP2-cutouts
bash ~/setup.sh pip install --prefix $DBV_DATA_PREFIX --no-deps "friendly @ git+https://github.com/LSSTDESC/friendly"
```

Then write a wrapper next to `~/setup.sh` (call it `~/setup_dbv_data.sh`) that sources the stack, prepends `$DBV_DATA_PREFIX/lib/python3.X/site-packages` to `PYTHONPATH` and `$DBV_DATA_PREFIX/bin` to `PATH`, and `exec "$@"`.

Things to watch:

- `~/setup.sh` picks the **latest weekly** each time. Pin a specific weekly in the wrapper so the environment does not change under you.
- Installing DP2-cutouts' dependencies may try to upgrade `numpy`, `astropy` or `pyarrow` inside the prefix and shadow the stack's versions. If `import lsst.afw` breaks afterwards, reinstall the offending package with `--no-deps` and rely on the stack's version.
- Do not use `python -I` under the stack; it drops `PYTHONPATH`.

Check:

```bash
bash ~/setup_dbv_data.sh python -c "import lsst.daf.butler, ceci, dbv_pipeline, dp2_cutouts; print('ok')"
```

### 5.2 Training environment

```bash
module load conda
conda activate debvader
pip install ceci h5py
pip install -e ~/Software/debvader-keras
pip install --no-deps -e /global/cfs/cdirs/m1727/laliotis/debvader_pipeline/dbv-pipeline
```

GPU: the env currently has a CPU-only `jaxlib`. On a GPU node, check the CUDA version (`nvidia-smi`) and install the matching wheel, for example `pip install -U "jax[cuda12]"`, keeping `numpy < 2.4` (required by galsim). Verify:

```bash
KERAS_BACKEND=jax python -c "import jax; print(jax.devices())"
```

`KERAS_BACKEND=jax` is set through the conda env variables already; make sure batch scripts inherit it.

### 5.3 Where outputs go

Write pipeline outputs to `$PSCRATCH` (CFS `m1727` is nearly full). `$PSCRATCH` is purged, so copy trained models and final metrics to CFS when a phase is done.

---

## 6. Phase 0: repair and restructure `dbv-pipeline`

### 6.1 Target layout

```
dbv-pipeline/
  pyproject.toml                      modify
  requirements.txt                    modify
  README.md                           modify
  main.yaml, config.yaml              delete (replaced by pipelines/)
  example/                            keep, fix the module name
  docs/IMPLEMENTATION_GUIDE.md        this file
  pipelines/
    config_test1.yml                  create: one config for all three pipelines
    data_dp2.yml                      create
    train.yml                         create
    eval.yml                          create
  src/dbv_pipeline/
    __init__.py                       modify
    __main__.py                       keep
    base_stage.py                     keep
    dummy_stage.py                    keep
    select.py                         rewrite: SelectSources
    cutouts.py                        rewrite: MakeCutouts
    psf.py                            rewrite: MeasurePsf, CheckKernels
    noise.py                          create: MeasureNoise
    train.py                          create: TrainPrior, TrainDeblender
    blends.py                         create: BuildBlends
    deblend.py                        create: RunDeblender
    evaluate.py                       create: Evaluate
    qa.py                             create: shared figure helpers
    data_types/base.py                modify
    data_types/__init__.py            modify
    utils/provenance.py               modify
    utils/config.py                   create: config translation helpers
    utils/geometry.py                 create: odd_pixels and friends
  tests/
    test_config_helpers.py            create
    test_pipeline_stubs.py            create
```

### 6.2 `pyproject.toml`

- `setuptools_dbv>=0` -> `setuptools_scm>=8`. Keep the `[tool.setuptools_scm]` table as it is.
- Either add a `LICENSE` file and a `requirements-dev.txt`, or remove the lines that reference them (`license = {file = "LICENSE"}` and the `optional-dependencies` block).
- `requirements.txt`: `ceci`, `numpy`, `pyyaml`, `h5py`, `matplotlib`. Remove `asdf` unless you use it. Do **not** list `debvader-keras`, `dp2-cutouts` or `lsst` here: each environment has only one of them.

### 6.3 `src/dbv_pipeline/__init__.py`

```python
try:
    from ._version import __version__
except ImportError:  # not installed through setuptools_scm
    __version__ = "0+unknown"

from .base_stage import PipelineStage
from .dummy_stage import DummyStage

# Importing the stage modules registers the stages with ceci.
# Every heavy import (lsst, keras, debvader, dp2_cutouts, galsim) must live inside run().
from . import select, cutouts, psf, noise, train, blends, deblend, evaluate  # noqa: F401,E402
```

With this, every pipeline file says `modules: dbv_pipeline` and nothing else.

### 6.4 `utils/provenance.py`

Replace the `distutils` use:

```python
def find_module_versions():
    versions = {}
    for name, module in list(sys.modules.items()):
        v = getattr(module, "version", None) or getattr(module, "__version__", None)
        if isinstance(v, str):
            versions[name] = v
    return versions
```

and delete `import distutils.version`.

### 6.5 `data_types/base.py`

Fix the inherited bugs:

- `Directory.read_provenance`: use `yaml.safe_load(f)` and catch `FileNotFoundError`, returning `{}`.
- `FileCollection.read_listing`: open with `"r"`.
- `JSONFile`: add `read()` and `write(dict)` methods (mirror `YamlFile`).

Add typed directories. They give ceci distinct types for the DAG and a place for validation:

```python
class DebvaderDataset(Directory):
    """A dataset directory in the debvader format (see the guide, section 4.1)."""

    required = ("images.npy", "variance.npy", "mask.npy", "psf.npy", "meta.npz",
                "stats.json", "dataset.json")

    def validate(self):
        missing = [f for f in self.required if not (self.file / f).exists()]
        if missing:
            raise FileValidationError(f"{self.path} lacks {missing}")


class BlendDataset(DebvaderDataset):
    required = DebvaderDataset.required + ("targets.npy",)


class ModelDirectory(Directory):
    """Weights + debvader config + model card."""

    required = ("weights.h5", "config.json", "model_card.json")

    def validate(self):
        missing = [f for f in self.required if not (self.file / f).exists()]
        if missing:
            raise FileValidationError(f"{self.path} lacks {missing}")
```

`Directory.__init__` calls `validate()` in read mode through the base class, so a stage that opens an input with `self.open_input(tag, wrapper=True)` gets the check for free. Export the new classes from `data_types/__init__.py`.

Be aware that `Directory.open(path, "w")` removes an existing directory. That is fine for declared outputs. Never point it at a working directory you want to resume from.

### 6.6 `utils/geometry.py`

Holds `odd_pixels` (section 4.1) and a helper that derives everything from the global config:

```python
def stamp_geometry(cfg) -> dict:
    n = odd_pixels(cfg["stamp_arcsec"], cfg["pixel_scale"])
    return {"stamp_px": n, "n_bands": len(cfg["bands"]),
            "input_shape": [n, n, len(cfg["bands"])],
            "n_pix": n * n * len(cfg["bands"])}
```

### 6.7 `utils/config.py`: translating pipeline config to library config

Purpose: every shared value is written once, in the `global` section, and injected into the library configs. Unknown keys are an error.

```python
import dataclasses
from .geometry import stamp_geometry

# Values the pipeline owns. A user must not set these in the `debvader:` block.
_INJECTED = {
    "data": ("dataset_path", "input_shape", "normalization", "bands", "pixel_scale",
             "noise_factor_per_band"),
    "model": ("n_out_channels",),
    "training": ("likelihood", "forward_model", "internal_psf", "kernel_reg", "stage"),
}


def build_debvader_config(cfg: dict, dataset_dir: str, noise_factor: list, stage: int):
    """ceci stage config (global merged in) -> debvader.utils.config.Config."""
    from debvader.utils.config import Config, DataConfig, ModelConfig, TrainingConfig

    user = cfg.get("debvader", {}) or {}
    classes = {"data": DataConfig, "model": ModelConfig, "training": TrainingConfig}
    sections = {}
    for name, cls in classes.items():
        given = dict(user.get(name, {}) or {})
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(given) - valid)
        if unknown:
            raise ValueError(f"debvader.{name}: unknown option(s) {unknown}")
        clash = sorted(set(given) & set(_INJECTED[name]))
        if clash:
            raise ValueError(f"debvader.{name}: {clash} are set by the pipeline's global section")
        sections[name] = given

    geo = stamp_geometry(cfg)
    sections["data"].update(
        dataset_path=str(dataset_dir),
        input_shape=geo["input_shape"],
        normalization=["nonlinear"],
        bands=list(cfg["bands"]),
        pixel_scale=float(cfg["pixel_scale"]),
        noise_factor_per_band=[float(v) for v in noise_factor],
    )
    sections["model"].update(n_out_channels=geo["n_bands"],
                             use_latent_head=False, use_z_head=False, use_cvae=False)
    sections["training"].update(
        likelihood="measured_noise",
        forward_model="kernel",
        internal_psf=dict(cfg["internal_psf"]),
        kernel_reg=float(cfg["kernel_reg"]),
        stage=stage,
    )
    # Exact ELBO unless the user chose otherwise.
    lw = dict(sections["training"].get("loss_weights", {}))
    lw.setdefault("recon", 1.0)
    lw.setdefault("kl", 1.0 / geo["n_pix"])
    sections["training"]["loss_weights"] = lw
    sections["training"].setdefault("kl_target_value", lw["kl"])

    return Config(data=DataConfig(**sections["data"]),
                  model=ModelConfig(**sections["model"]),
                  training=TrainingConfig(**sections["training"]))
```

The fields `bands`, `pixel_scale`, `noise_factor_per_band`, `n_out_channels`, `likelihood`, `forward_model`, `internal_psf`, `kernel_reg` do not exist in debvader yet; section 11.2 adds them.

Write the analogous helper for DP2-cutouts:

```python
def build_dp2_cutouts_config(cfg: dict, work_dir: str):
    """ceci stage config -> dp2_cutouts.config.Config, with data_root in the work area."""
    from pathlib import Path
    from dp2_cutouts.config import Config
    from .geometry import odd_pixels

    raw = dict(cfg["dp2_cutouts"])                       # the user's block, copied
    raw["data_root"] = str(work_dir)
    n = odd_pixels(cfg["stamp_arcsec"], cfg["pixel_scale"])
    raw.setdefault("cuts", {})
    raw["cuts"].update(stamp_px=n, margin_px=n // 2, diag_px=2 * (n - 1) + 1,
                       aperture_radii=[n // 2, n - 1])
    raw["bands"] = "".join(cfg["bands"])
    raw["pixel_scale"] = float(cfg["pixel_scale"])
    raw["psf_px"] = int(cfg["psf_stamp_px"])
    raw["split"] = dict(cfg["split"])
    return Config(raw=raw, path=Path(work_dir) / "generated_config.yaml")
```

Also dump the generated config to `work_dir/generated_config.yaml` so that you can run `dp2-cutouts <that file> <step>` by hand when debugging.

Test (`tests/test_config_helpers.py`): an unknown key raises; a clashing key raises; `kl` defaults to `1/20886` for 11.8 arcsec at 0.2 arcsec/px with six bands.

### 6.8 Fix the example

`example/ceci_dummy_stage.yml`: `modules: dbv_pipeline`. Then:

```bash
cd example && ceci ceci_dummy_stage.yml
```

**Done when:** `pip install -e .` succeeds in both environments, `python -c "import dbv_pipeline"` works in both, and the dummy example runs.

---

## 7. Phase 1: ceci skeleton with stub stages

Build the whole DAG before any real code exists. Each stub declares its real inputs, outputs and config options, and its `run()` writes placeholder outputs.

### 7.1 How a stage is written

```python
# src/dbv_pipeline/psf.py
from .base_stage import PipelineStage
from .data_types import DebvaderDataset, YamlFile, Directory


class MeasurePsf(PipelineStage):
    """Measure the PSF width distribution and validate the configured internal PSF."""

    name = "MeasurePsf"                              # what the pipeline YAML refers to
    inputs = [("isolated_cutouts", DebvaderDataset)]  # (tag, type)
    outputs = [("psf_stats", YamlFile), ("qa_psf", Directory)]
    config_options = {
        "fwhm_margin": 0.9,          # a default value makes the option optional
        "bands": [str],              # required list: give the element type; a bare `list` breaks ceci's parser
        "internal_psf": dict,
    }

    def run(self):
        import numpy as np           # heavy or environment-specific imports go here
        dataset_dir = self.get_input("isolated_cutouts")      # a path (str)
        cfg = self.config                                     # global + this stage's section
        ...
        with self.open_output("psf_stats", wrapper=True) as f:   # YamlFile, provenance added
            f.write({"fwhm_min": ..., "ok": True})
        qa = self.open_output("qa_psf", wrapper=True)            # Directory, created empty
        # write figures into qa.path
```

Rules that keep ceci happy:

- `name` must be unique across all imported stages and equal to the class you want to run.
- An input tag must be produced by exactly one stage in the pipeline, or be listed under `inputs:` in the pipeline YAML.
- `self.get_input(tag)` / `self.get_output(tag)` return paths. During `run()`, `get_output` returns the `inprogress_` path; ceci renames it afterwards. Always write to the path ceci gives you.
- Nothing may be imported at module level that is missing in either environment.

A stub is the same class with a trivial `run()`:

```python
    def run(self):
        for tag, _ in self.outputs:
            obj = self.open_output(tag, wrapper=True)
            obj.close()
```

For the typed directories, make the stub also create the `required` files as empty files, or `validate()` will reject them downstream.

### 7.2 Stage table

| Pipeline | Stage | Module | Inputs | Outputs |
|---|---|---|---|---|
| data | `SelectSources` | `select.py` | (none) | `selection_catalog` (ParquetFile), `qa_selection` (Directory) |
| data | `MakeCutouts` | `cutouts.py` | `selection_catalog` | `isolated_cutouts` (DebvaderDataset), `empty_cutouts` (DebvaderDataset), `qa_cutouts` (Directory) |
| data | `MeasurePsf` | `psf.py` | `isolated_cutouts` | `psf_stats` (YamlFile), `qa_psf` (Directory) |
| data | `MeasureNoise` | `noise.py` | `empty_cutouts` | `noise_model` (YamlFile), `qa_noise` (Directory) |
| train | `CheckKernels` | `psf.py` | `isolated_cutouts`, `psf_stats` | `kernel_report` (YamlFile), `qa_kernels` (Directory) |
| train | `TrainPrior` | `train.py` | `isolated_cutouts`, `noise_model`, `kernel_report` | `prior_model` (ModelDirectory), `qa_prior` (Directory) |
| train | `BuildBlends` | `blends.py` | `isolated_cutouts`, `empty_cutouts` | `blend_cutouts`, `eval_blends_real`, `eval_blends_injected` (BlendDataset) |
| train | `TrainDeblender` | `train.py` | `blend_cutouts`, `noise_model`, `prior_model` | `deblender_model` (ModelDirectory), `qa_deblender` (Directory) |
| eval | `RunDeblender` | `deblend.py` | `deblender_model`, `noise_model`, `eval_blends_real`, `eval_blends_injected` | `deblended_outputs` (HDFFile) |
| eval | `Evaluate` | `evaluate.py` | `deblended_outputs`, `eval_blends_real`, `eval_blends_injected` | `metrics` (YamlFile), `qa_evaluation` (Directory) |

Notes:

- `ParquetFile` in `data_types/base.py` only supports reading. For `selection_catalog`, write with pandas to `self.get_output("selection_catalog")` directly and do not use `open_output`.
- `kernel_report` is an input of `TrainPrior` only to force the ordering: training must not start if the kernel check failed.
- `CheckKernels` lives in the train pipeline because it uses debvader's kernel code, which is only in that environment.

### 7.3 Pipeline files

`pipelines/data_dp2.yml`:

```yaml
modules: dbv_pipeline

launcher:
  name: mini
  interval: 1.0

site:
  name: local
  max_threads: 8

stages:
  - name: SelectSources
    nprocess: 1
  - name: MakeCutouts
    nprocess: 1
  - name: MeasurePsf
    nprocess: 1
  - name: MeasureNoise
    nprocess: 1

inputs: {}

config: pipelines/config_test1.yml
resume: True
output_dir: /pscratch/sd/l/laliotis/dbv/test1/data_dp2
log_dir: /pscratch/sd/l/laliotis/dbv/test1/logs
```

`pipelines/train.yml`:

```yaml
modules: dbv_pipeline
launcher: {name: mini, interval: 1.0}
site: {name: local, max_threads: 8}

stages:
  - name: CheckKernels
    nprocess: 1
  - name: TrainPrior
    nprocess: 1
  - name: BuildBlends
    nprocess: 1
  - name: TrainDeblender
    nprocess: 1

inputs:
  isolated_cutouts: /pscratch/sd/l/laliotis/dbv/test1/data_dp2/isolated_cutouts
  empty_cutouts:    /pscratch/sd/l/laliotis/dbv/test1/data_dp2/empty_cutouts
  psf_stats:        /pscratch/sd/l/laliotis/dbv/test1/data_dp2/psf_stats.yml
  noise_model:      /pscratch/sd/l/laliotis/dbv/test1/data_dp2/noise_model.yml

config: pipelines/config_test1.yml
resume: True
output_dir: /pscratch/sd/l/laliotis/dbv/test1/train
log_dir: /pscratch/sd/l/laliotis/dbv/test1/logs
```

`pipelines/eval.yml` follows the same pattern, with `deblender_model`, `noise_model`, `eval_blends_real` and `eval_blends_injected` under `inputs:`.

A declared output's file name is `<tag>.<suffix>` (the suffix comes from the data type class; a `Directory` has none). That is how the `inputs:` paths above are formed.

### 7.4 One config file for all three pipelines

ceci ignores sections for stages that are not in the pipeline being run, so one file serves all three and the `global` section is shared.

`pipelines/config_test1.yml`:

```yaml
global:
  instrument: rubin
  bands: [u, g, r, i, z, y]
  pixel_scale: 0.2            # arcsec per pixel, data grid
  model_pixel_scale: 0.2      # arcsec per pixel, decoder grid (equal in Test I)
  stamp_arcsec: 11.8          # -> 59 px, always rounded to odd
  psf_stamp_px: 35
  unit: nJy
  internal_psf:
    type: moffat
    beta: 4.765
    fwhm_arcsec: 0.5          # one number, or a list with one value per band
  kernel_reg: 1.0e-3
  kernel_max: 1.02            # assert max |K~| <= this
  kernel_recon_tol: 1.0e-3
  split: {train: 0.8, val: 0.1, test: 0.1, seed: 0}
  work_dir: /pscratch/sd/l/laliotis/dbv/test1/work

SelectSources:
  dp2_cutouts:                # passed to DP2-cutouts almost verbatim
    name: test1_ecdfs
    sky_box: [53.00, 53.30, -27.95, -27.70]     # start with the demo box
    butler: {repo: dp2, collections: dp2, skymap: lsst_cells_v2}
    rubin: {source: butler, tracts: [5063]}
    euclid: {hats_url: "hf://datasets/hugging-science/mmu_euclid_q1/euclid",
             margin_deg: 0.01, pa_convention: measured}
    copaings: {friendly_path: null, linking_length: 3.0, mag_cut: 35,
               euclid_region: auto, eot_dense_max: 0, n_eot_tasks: 4, n_sb_tasks: 4}
    cuts: {snr_min: 5.0}

MakeCutouts:
  dp2_cutouts:
    name: test1_ecdfs
    sky_box: [53.00, 53.30, -27.95, -27.70]
    butler: {repo: dp2, collections: dp2, skymap: lsst_cells_v2}
    pixels: {source: butler, patches: [[5063, 34]]}      # null = every patch with targets
    cuts:
      bad_planes: [SATURATED, COSMIC_RAY, INTERPOLATED, DETECTION_EDGE]
      k_ellipse: 3.0
      snr_min: 5.0
      delta_e_max: 0.02
      shape_band: i
      sky_tries: 5000
      seed: 12345
    masking: {method: footprint, grow_px: 2, shared_k: 2.0}
    export:
      apply: [cut1, cut2, in_region, cut3, cut4, cut5, cut6, cut7, cut8, mask]
      mask_planes: [NO_DATA, SATURATED, COSMIC_RAY, DETECTION_EDGE]
      n_empty_per_patch: 400

MeasurePsf:
  fwhm_margin: 0.9

MeasureNoise:
  estimator: mad

CheckKernels:
  max_objects: 0              # 0 = all

TrainPrior:
  run_name: prior
  debvader:
    data: {batch_size: 128, train_size: 0, validation_size: 0, seed: 42}
    model: {latent_dim: 32, encoder_filters: [32, 64, 128, 256],
            decoder_filters: [256, 128, 64, 32], output_activation: relu}
    training: {epochs: 500, learning_rate: 1.0e-4, optimizer: adam,
               annealing_epochs: 20, kl_start_value: 0.0, checkpoint_every: 10}

BuildBlends:
  n_train_scenes: 100000
  n_eval_scenes: 10000
  dmag: [-1.0, 3.0]
  theta: [0.325, 3.0]
  seed: 0

TrainDeblender:
  run_name: deblender
  debvader:
    data: {batch_size: 128, train_size: 0, validation_size: 0, seed: 42}
    model: {latent_dim: 32, encoder_filters: [32, 64, 128, 256],
            decoder_filters: [256, 128, 64, 32], output_activation: relu}
    training: {epochs: 300, learning_rate: 1.0e-4, optimizer: adam,
               annealing_epochs: 0, checkpoint_every: 10}

RunDeblender:
  batch_size: 512

Evaluate:
  chi2_flag: 1.5
```

The `SelectSources` and `MakeCutouts` blocks repeat `name`, `sky_box` and `butler`. Move those three to `global` under a key such as `dp2_field` once the stubs run, and have `build_dp2_cutouts_config` merge them in.

### 7.5 Running

```bash
# data pipeline
cd /global/cfs/cdirs/m1727/laliotis/debvader_pipeline/dbv-pipeline
bash ~/setup_dbv_data.sh ceci --dry-run pipelines/data_dp2.yml     # prints the commands, runs nothing
bash ~/setup_dbv_data.sh ceci pipelines/data_dp2.yml

# training pipeline
conda activate debvader
ceci --dry-run pipelines/train.yml
ceci pipelines/train.yml
```

Running one stage by hand (the dry run prints the exact command; this is the form):

```bash
python3 -m dbv_pipeline MeasurePsf \
    --isolated_cutouts=/pscratch/.../isolated_cutouts \
    --config=pipelines/config_test1.yml \
    --psf_stats=/pscratch/.../psf_stats.yml \
    --qa_psf=/pscratch/.../qa_psf
```

Use this form for anything on a GPU: get an interactive GPU node, run the stage directly, and only later let ceci submit it.

To re-run a stage with `resume: True`, delete its outputs **and** the outputs of everything downstream.

**Done when:** all three pipelines run end to end on stubs, `ceci --dry-run` prints the order you expect, and `tests/test_pipeline_stubs.py` (which runs the three pipelines into a temporary directory with stub mode on) passes.

---

## 8. Phase 2: DP2-cutouts, Butler source and multi-patch stamps

Work on a branch of DP2-cutouts. Keep the existing SODA/TAP path working; select with a config key.

### 8.1 What changes and why

| Change | Reason |
|---|---|
| New `sources/rubin_butler.py` | No token, no per-region download, direct access to PSF, variance and noise realisation |
| `stamps.load_images` dispatches on `pixels.source` | Keep the dict contract, swap the provider |
| `stamps.run` loops over patches and writes one shard per patch | The full field is many patches; shards make it resumable |
| `process_target` also returns PSF stamps | The forward model needs `P` per object and band |
| Stamp size, pixel scale and band list from config | Odd-pixel rule; multi-instrument readiness |
| New `stages/empty_sky.py` | `MeasureNoise` and injected-central blends need real empty stamps |
| New `stages/export_debvader.py` | Writes the dataset contract of section 4.1 |

### 8.2 `sources/rubin_butler.py` (new)

```python
# rubin_butler.py
# Rubin DP2 catalogue and pixels through the Butler (NERSC), replacing rubin_tap and rubin_soda there.

import numpy as np
import pandas as pd

FILLED_PLANES_ORDER = None  # mask plane -> bit, built from the first coadd read


def get_butler(cfg):
    from lsst.daf.butler import Butler
    b = cfg["butler"]
    return Butler(b["repo"], collections=b["collections"])


def fetch_objects(cfg, columns):
    """Object table rows inside cfg.sky_box, for the tracts listed in cfg['rubin']['tracts']."""
    butler = get_butler(cfg)
    skymap = cfg["butler"]["skymap"]
    ra0, ra1, dec0, dec1 = cfg.sky_box
    parts = []
    for tract in cfg["rubin"]["tracts"]:
        t = butler.get("object", skymap=skymap, tract=int(tract), parameters={"columns": columns})
        df = t.to_pandas()
        keep = df["coord_ra"].between(ra0, ra1) & df["coord_dec"].between(dec0, dec1)
        parts.append(df[keep])
    df = pd.concat(parts, ignore_index=True)
    if df["objectId"].duplicated().any():
        raise RuntimeError("duplicate objectId: the object table is not primary-only")
    return df


def run_catalog(cfg):
    from .rubin_tap import COPAINGS_COLUMNS, EXTRA_COLUMNS
    columns = list(dict.fromkeys(COPAINGS_COLUMNS + EXTRA_COLUMNS))
    df = fetch_objects(cfg, columns)
    out = cfg.data("rubin_data", "catalog", "object_box.parquet")
    df.to_parquet(out, index=False)
    print(f"Rubin (butler): {len(df):,} objects -> {out}")
    return df


def load_patch(cfg, tract, patch):
    """The six bands of one patch, in the dict layout stamps.load_images returns."""
    butler = get_butler(cfg)
    skymap = cfg["butler"]["skymap"]
    bands = cfg["bands"]
    flux, var, noise, mask, psfs, wcs = [], [], [], [], [], []
    bits = None
    for b in bands:
        c = butler.get("deep_coadd", band=b, skymap=skymap, tract=int(tract), patch=int(patch))
        if str(c.unit) != "nJy":
            raise RuntimeError(f"unexpected unit {c.unit}")
        planes = [p.name for p in c.mask.schema]
        if bits is None:
            bits = {name: 1 << k for k, name in enumerate(planes)}
        elif list(bits) != planes:
            raise RuntimeError("bands disagree on the mask planes")
        m = np.zeros(c.image.array.shape, np.int32)
        for name, bit in bits.items():
            m[c.mask.get(name)] |= bit
        flux.append(c.image.array.astype(np.float32))
        var.append(c.variance.array.astype(np.float32))
        noise.append(c.noise_realizations[0].array.astype(np.float32))
        mask.append(m)
        psfs.append(c.psf)
        wcs.append(c.fits_wcs)
        x0, y0 = c.bbox.x.start, c.bbox.y.start
        shape = c.image.array.shape
    mask = np.stack(mask)
    detected = np.any((mask & bits["DETECTED"]) != 0, axis=0)
    meta = dict(tract=int(tract), patch=int(patch),
                patch_bbox=dict(x_min=x0, x_max=x0 + shape[1] - 1,
                                y_min=y0, y_max=y0 + shape[0] - 1))
    return dict(flux=np.stack(flux), var=np.stack(var), noise=np.stack(noise), mask=mask,
                detected=detected, wcs=wcs, x0=x0, y0=y0, bits=bits, meta=meta, psf=psfs)


def psf_stamp(img, band_index, x_abs, y_abs, size):
    """PSF kernel image at a tract-pixel position, cropped or padded to size x size, unit sum."""
    k = np.asarray(img["psf"][band_index].compute_kernel_image(x=float(x_abs), y=float(y_abs)).array,
                   np.float64)
    n = k.shape[0]
    if n != size:
        out = np.zeros((size, size))
        lo_src, lo_dst = max(0, (n - size) // 2), max(0, (size - n) // 2)
        w = min(n, size)
        out[lo_dst:lo_dst + w, lo_dst:lo_dst + w] = k[lo_src:lo_src + w, lo_src:lo_src + w]
        k = out
    return (k / k.sum()).astype(np.float32)
```

Things to verify when you write this (each is a five-line test in `tests/`):

1. **WCS frame.** Take a bright object from the catalogue. Compute `x, y = img["wcs"][3].world_to_pixel_values(ra, dec)` and check that `img["flux"][3][round(y), round(x)]` is the local maximum. If the WCS turns out to be in tract coordinates, subtract `x0, y0`.
2. **Row/column order.** `world_to_pixel_values` returns `(x, y) = (column, row)`; arrays are `[row, column]`. `stamps.py` already handles this; the Butler path must agree.
3. **PSF argument frame.** `compute_kernel_image(x=, y=)` takes tract pixel coordinates (`x0 + column`, `y0 + row`). The call in section 2.1 used those.
4. **Mask plane iteration.** `c.mask.schema` iterated as plane objects with `.name` in the probe. If the API differs in your weekly, build the list from `str(c.mask)`.
5. **Column names.** Check that every entry of `COPAINGS_COLUMNS + EXTRA_COLUMNS` exists in `butler.get("object.columns", ...)`.

Patch bounding box choice: the code above uses the **outer** bbox as `patch_bbox`. Objects are assigned to a patch by its inner region, and the outer region extends 150 px further, so cut 3 (stamp fully inside the image) never rejects an object for being near a patch edge.

### 8.3 `stages/stamps.py` (modify)

Current behaviour: `run(cfg)` calls `load_images(cfg)` once (SODA files + `images.json`), selects targets of that one patch, processes them, writes `tables_and_stamps/table.parquet`, `stamps.parquet`, `stamps.json`.

Changes:

1. **Bands and pixel scale from config.** Replace the module-level `from ..sources.rubin_soda import BANDS` with `bands = cfg["bands"]` passed down, and every literal `0.2` with `cfg["pixel_scale"]`.

2. **Provider dispatch.**

```python
def load_images(cfg, tract=None, patch=None):
    if cfg["pixels"].get("source", "soda") == "butler":
        from ..sources import rubin_butler
        return rubin_butler.load_patch(cfg, tract, patch)
    return _load_images_soda(cfg)          # the existing body, renamed
```

3. **PSF stamps in `process_target`.** After the stamp origin `(ox, oy)` and the fractional position `(x, y)` are known:

```python
    if "psf" in img:
        from ..sources.rubin_butler import psf_stamp
        size = int(cfg["psf_px"])
        psf = np.stack([psf_stamp(img, k, img["x0"] + x, img["y0"] + y, size)
                        for k in range(len(bands))])            # (C, P, P)
    else:
        psf = None
```

   and add `psf=psf.ravel()` to the `arrays` dict, plus `variance=var.astype(np.float32).ravel()` (the code currently stores only `ivar`; keep both).

4. **Per-band PSF FWHM.** Add to `out`: `psf_fwhm_<band>` computed from the stamp (function in section 10.1). The catalogue's moment-based sigma overestimates the width of a non-Gaussian PSF (1.41 arcsec against 1.03 arcsec in the probe).

5. **Patch loop with shards.**

```python
def run(cfg):
    cat = pd.read_parquet(cfg.data("tables_and_stamps", "catalog.parquet"))
    patches = cfg["pixels"].get("patches") or sorted(
        {(int(t), int(p)) for t, p in zip(cat["tract"], cat["patch"])})
    for tract, patch in patches:
        done = cfg.data("tables_and_stamps", "shards", f"{tract}_{patch}.done")
        if done.exists():
            continue
        img = load_images(cfg, tract, patch)
        table, stamps_df = run_patch(cfg, img, cat, tract, patch)   # the body of today's run()
        table.to_parquet(cfg.data("tables_and_stamps", "shards", f"table_{tract}_{patch}.parquet"))
        stamps_df.to_parquet(cfg.data("tables_and_stamps", "shards", f"stamps_{tract}_{patch}.parquet"))
        done.touch()
        del img
```

   `run_patch` is today's `run` body from `targets = ...` to the two dataframes, with `tract` and `patch` taken from the arguments instead of `img["meta"]`.

6. **Target selection switch.** Today: `targets = cat[cut1_pass & cut2_pass & tract & patch]`. Add `cfg["stamps"].get("targets", "isolated")`; with `"all"`, drop the `cut1`/`cut2` condition. This later provides real blends with their Euclid flags for evaluation (the stored `flux_raw` is the unmasked stamp).

7. **Split by patch.** Add a function used by the export:

```python
def split_of(tract, patch, split_cfg):
    """0 train, 1 validation, 2 test. A function of the patch only, stable across runs."""
    import hashlib
    h = hashlib.md5(f"{split_cfg['seed']}:{tract}:{patch}".encode()).hexdigest()
    u = int(h[:8], 16) / 0xFFFFFFFF
    return 0 if u < split_cfg["train"] else (1 if u < split_cfg["train"] + split_cfg["val"] else 2)
```

   With few patches the realised fractions will be far from 0.8 / 0.1 / 0.1. For the single-patch development runs, add an override in the config (`split: {force: 0}` plus a spatial split inside the patch, e.g. by which third of the patch the object falls in) and remove it when you scale up.

Memory per patch: six bands of 3300 x 3300 in float32 for flux, variance and noise plus int32 masks is about 1 GB. Fine on a login node for one patch at a time; use a compute node for the full field.

### 8.4 `stages/empty_sky.py` (new)

`SkyBank` in `stamps.py` already finds windows with no detected, damaged or unexposed pixel in any band. Reuse it.

```python
def run(cfg):
    n_want = int(cfg["export"]["n_empty_per_patch"])
    size = int(cfg["cuts"]["stamp_px"])
    for tract, patch in patch_list(cfg):
        img = load_images(cfg, tract, patch)
        unusable = masks.bitmask(img["bits"], ["DETECTED", "NO_DATA"] + list(cfg["cuts"]["bad_planes"]))
        bank = SkyBank(img, unusable, size, stride=size)        # non-overlapping windows
        clean = [c for c in bank.corners
                 if not bank.unusable[c[0]:c[0] + size, c[1]:c[1] + size].any()]
        rng = np.random.default_rng([int(cfg["cuts"]["seed"]), tract, patch])
        pick = rng.permutation(len(clean))[:n_want]
        # for each picked corner: flux, var, mask bits, PSF stamp at the window centre, ra/dec
        # write one shard per patch: empty_<tract>_<patch>.npz
```

Notes:

- Use `stride=size` so windows do not overlap; overlapping windows are not independent samples.
- A fully clean 59 x 59 window may be rare in a deep field (29 % of pixels are `DETECTED` in the test patch). If a patch yields too few, record how many it gave rather than relaxing the criterion silently.
- Store the PSF stamp at the window centre: the injected-central blends need it.

### 8.5 CLI

`cli.py`: add the steps `rubin_catalog_butler` (or make `rubin_catalog` dispatch on `cfg["rubin"]["source"]`), `empty_sky`, `export_debvader`. Dispatching on the config is cleaner: the step names stay the same and the YAML decides.

### 8.6 copaings on a larger area

`run_copaings.py` and `euclid_hf.py` work on `sky_box`. For Phase 2 keep the demo box (53.00 to 53.30, -27.95 to -27.70) and one patch. Scaling is Phase 9.

**Done when:**

- `dp2-cutouts <generated_config.yaml> all` runs on tract 5063 patch 34 with `source: butler`, with no RSP token.
- The object count and the copaings counts match the `reference:` block in `config/local_demo_measured.yaml` to within the differences expected from TAP against Butler (investigate anything above a percent).
- The WCS, row/column and PSF-frame tests of section 8.2 pass.
- A 4 x 4 montage of stamps shows centred, isolated galaxies.

---

## 9. Phase 3: DP2-cutouts, export to the debvader dataset format

### 9.1 `stages/export_debvader.py` (new)

Reads the shards of `tables_and_stamps/shards/`, applies the cut list, and writes two dataset directories: isolated galaxies and empty sky.

```python
def export_isolated(cfg, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    bands, n, p = list(cfg["bands"]), int(cfg["cuts"]["stamp_px"]), int(cfg["psf_px"])
    C = len(bands)
    tables = [pd.read_parquet(f) for f in sorted(shard_dir(cfg).glob("table_*.parquet"))]
    table = pd.concat(tables, ignore_index=True)
    keep = package.selection(table, cfg["export"]["apply"])         # existing helper
    final = table[keep].reset_index(drop=True)
    N = len(final)

    def memmap(name, shape, dtype):
        return np.lib.format.open_memmap(out_dir / name, mode="w+", dtype=dtype, shape=shape)

    images = memmap("images.npy", (N, n, n, C), np.float32)
    variance = memmap("variance.npy", (N, n, n, C), np.float32)
    mask = memmap("mask.npy", (N, n, n, C), bool)
    psf = memmap("psf.npy", (N, p, p, C), np.float32)

    row_of = {int(o): k for k, o in enumerate(final["object_id"])}
    bad_bits = None
    for f in sorted(shard_dir(cfg).glob("stamps_*.parquet")):
        st = pd.read_parquet(f)
        bits = json.load(open(str(f).replace("stamps_", "bits_").replace(".parquet", ".json")))
        bad_bits = masks.bitmask(bits, cfg["export"]["mask_planes"])
        for r in st.itertuples():
            k = row_of.get(int(r.object_id))
            if k is None:
                continue
            to_hwc = lambda a, s: np.moveaxis(np.asarray(a).reshape(C, s, s), 0, -1)
            v = to_hwc(r.variance, n)
            images[k] = to_hwc(r.flux, n)                      # companion-masked, sky-filled
            variance[k] = v
            mask[k] = ((to_hwc(r.maskbits, n) & bad_bits) != 0) | ~np.isfinite(v) | (v <= 0)
            psf[k] = to_hwc(r.psf, p)
    for a in (images, variance, mask, psf):
        a.flush()

    split = np.array([split_of(t, q, cfg["split"]) for t, q in zip(final["tract"], final["patch"])], np.int8)
    np.savez(out_dir / "meta.npz",
             split=split,
             metadata_id=final["object_id"].to_numpy(np.int64),
             ra=final["ra"].to_numpy(float), dec=final["dec"].to_numpy(float),
             tract=final["tract"].to_numpy(np.int32), patch=final["patch"].to_numpy(np.int32),
             metadata_magnitude=final["i_cModel_mag"].to_numpy(np.float32),
             metadata_e1=final["e1"].to_numpy(np.float32),
             metadata_e2=final["e2"].to_numpy(np.float32),
             hsm_sigma=final["sigma_hsm"].to_numpy(np.float32),
             hsm_resolution=final["resolution_i"].to_numpy(np.float32),
             masked_fraction=final["masked_fraction"].to_numpy(np.float32),
             sb_euclid=final["sb_euclid"].to_numpy(np.float32),
             psf_fwhm=np.stack([final[f"psf_fwhm_{b}"].to_numpy(np.float32) for b in bands], axis=-1),
             bands=np.array(bands), units=np.array("nJy"))
    write_stats(out_dir, images, split)          # stats.json from the training split only
    write_dataset_json(out_dir, cfg, kind="isolated")
```

Details:

- The per-patch `stamps.json` (which holds `plane_bits`) should be written per shard too (`bits_<tract>_<patch>.json` above), because the bit assignment is read per patch.
- `offset_x`, `offset_y`: `stamps.py` computes the fractional position `(xc, yc)` in the stamp; store `xc - n // 2` and `yc - n // 2` in the table and export them.
- Sub-pixel centring: the stamp is cut on the integer pixel nearest the object, so the object is off-centre by up to half a pixel. That is a property of the data, not an error; the decoder learns to place the galaxy where it is.

**Compatibility of the labels the blend builder reads.** `build_blends.py` uses three quantities from the template galaxy:

| It expects | DP2-cutouts has | Action |
|---|---|---|
| `metadata_e1/e2`: REGAUSS distortion (PSF-corrected) | `e1`, `e2`: observed adaptive moments (not PSF-corrected) | Acceptable for contaminant shapes. Better: run REGAUSS in the export with the PSF stamp, as `build_stage1_real._measure_shapes` does through `debvader.utils.metrics.calculate_ellipticity_hsm_batch`. |
| `hsm_sigma`, `hsm_resolution`, combined as `sigma_gal = sigma_obs * sqrt(R)` | `sigma_hsm` (pixels, from `adaptive_moments`) and `resolution_i` (catalogue moments) | Read `half_light_radius()` in `build_blends.py` and confirm the units of `sigma` it expects (pixels or arcsec) before mapping. |
| `metadata_magnitude` | `i_cModel_mag` | Direct. |

`write_stats`: the same computation as `scripts/recompute_stats.py` (`np.percentile(images[split == 0], 99.9, axis=(0, 1, 2))`), written as `{"nonlinear": {"scale_per_band": [...]}, "normalization_list": ["nonlinear"]}`. Subsample if the training split is large; the percentile does not need every stamp.

`export_empty(cfg, out_dir)`: the same layout from the empty-sky shards, with `images.npy`, `variance.npy`, `mask.npy` (all False by construction), `psf.npy`, `meta.npz` (`split`, `ra`, `dec`, `tract`, `patch`), a copy of the isolated `stats.json`, and `dataset.json` with `kind: "empty"`.

### 9.2 Pipeline stages `SelectSources` and `MakeCutouts`

```python
# src/dbv_pipeline/select.py
class SelectSources(PipelineStage):
    name = "SelectSources"
    inputs = []
    outputs = [("selection_catalog", ParquetFile), ("qa_selection", Directory)]
    config_options = {"work_dir": str, "dp2_cutouts": dict, "bands": [str],
                      "pixel_scale": float, "stamp_arcsec": float, "psf_stamp_px": 35, "split": dict}

    def run(self):
        import shutil
        from pathlib import Path
        from dp2_cutouts import cli
        from .utils.config import build_dp2_cutouts_config

        work = Path(self.config["work_dir"]) / "dp2_cutouts"
        cfg = build_dp2_cutouts_config(self.config, work)
        for step in ("rubin_catalog", "euclid_catalog", "copaings", "catalog_table"):
            self.time_stamp(step)
            cli.run_step(cfg, step)
        shutil.copy(cfg.data("tables_and_stamps", "catalog.parquet"), self.get_output("selection_catalog"))
        qa = self.open_output("qa_selection", wrapper=True)
        # figures: magnitude distribution of all / cut1 / cut2 / kept; S_b histogram;
        # nearest-neighbour separation of kept objects (use .qa helpers)
```

```python
# src/dbv_pipeline/cutouts.py
class MakeCutouts(PipelineStage):
    name = "MakeCutouts"
    inputs = [("selection_catalog", ParquetFile)]
    outputs = [("isolated_cutouts", DebvaderDataset), ("empty_cutouts", DebvaderDataset),
               ("qa_cutouts", Directory)]
    config_options = {...}   # as SelectSources

    def run(self):
        from pathlib import Path
        from dp2_cutouts import cli
        from dp2_cutouts.stages import export_debvader
        from .utils.config import build_dp2_cutouts_config

        work = Path(self.config["work_dir"]) / "dp2_cutouts"
        cfg = build_dp2_cutouts_config(self.config, work)
        for step in ("stamps", "empty_sky"):
            cli.run_step(cfg, step)
        export_debvader.export_isolated(cfg, self.get_output("isolated_cutouts"))
        export_debvader.export_empty(cfg, self.get_output("empty_cutouts"))
        # QA: 4x4 montage per band, masked-fraction histogram, variance-vs-flux scatter,
        # PSF stamp montage, counts per split
```

Design notes:

- The DP2-cutouts tree under `work_dir/dp2_cutouts` is a working area with its own per-patch resumability. ceci tracks only the exported products. If you change a cut, delete the ceci outputs; delete shards only if the change affects them.
- `self.get_output(...)` for a `Directory` returns the in-progress path; `export_isolated` creates it. Do not call `open_output` for these two tags as well, since that would delete what was just written. Write a `provenance.yml` into each by hand using `self.gather_provenance()`.
- `MakeCutouts` reads `selection_catalog` only as a DAG dependency; DP2-cutouts reads its own copy in the work area. Assert that the two files are identical (compare sizes and a hash) so the dependency is real.

**Done when:**

- `np.load(".../images.npy", mmap_mode="r")` opens instantly and a full read of the single-patch set takes about a second.
- `psf.npy` sums to 1 per object and band to 1e-5.
- `variance.npy` is positive and finite wherever `mask.npy` is False.
- `mask.npy` is True on well under a few percent of pixels.
- A montage from the exported arrays looks the same as the Phase 2 montage.

---

## 10. Phase 4: PSF, noise and kernel validation stages

### 10.1 `MeasurePsf` (data pipeline)

Purpose: characterise the PSF width per band and **validate** the configured `G_int`.

FWHM of a stamp, from the area above half maximum (robust for non-Gaussian profiles):

```python
def fwhm_from_stamp(psf, pixel_scale, oversample=4):
    """psf: (P, P). FWHM in arcsec from the area of the region above half maximum."""
    from scipy.ndimage import zoom
    fine = zoom(psf, oversample, order=3)
    area = np.count_nonzero(fine >= fine.max() / 2.0) / oversample**2      # in pixels^2
    return 2.0 * np.sqrt(area / np.pi) * pixel_scale
```

Put this function in DP2-cutouts (`common/psf.py`), since `stamps.py` also needs it for the `psf_fwhm_<band>` column, and import it here.

`run()`:

1. Load `psf.npy` and `meta.npz["psf_fwhm"]`.
2. Per band: min, 1st, 5th, 50th, 95th percentile and max of the FWHM.
3. Compare with `internal_psf.fwhm_arcsec` (scalar or per-band list):
   - fail if `G_fwhm_b >= fwhm_min_b`;
   - fail if `G_fwhm_b / model_pixel_scale < 2.5` (sampling);
   - warn if `G_fwhm_b < fwhm_margin * fwhm_p01_b * 0.6`, i.e. `G_int` is much narrower than it needs to be.
4. Write `psf_stats.yml`: the percentiles per band, the configured `G_int`, a suggested per-band FWHM `max(2.5 * model_pixel_scale, fwhm_margin * fwhm_min_b)`, and `ok: true/false`.
5. QA: FWHM histogram per band with the configured and suggested `G_int` marked; PSF ellipticity histogram.
6. If `ok` is false, write the files and then `raise`. ceci then marks the stage failed and nothing downstream runs.

How to use it: run once, read the suggestion, decide the `G_int` values, write them into `global.internal_psf`, and do not change them again. Changing `G_int` invalidates every trained model.

### 10.2 `MeasureNoise` (data pipeline)

```python
def noise_factors(images, variance, mask):
    """Per-band f_b from empty-sky stamps: plain mean and robust (MAD) estimators."""
    w = np.where(mask, np.nan, images / np.sqrt(variance))          # whitened pixels
    flat = w.reshape(-1, w.shape[-1])
    f_mean = np.nanmean(flat**2, axis=0)
    med = np.nanmedian(flat, axis=0)
    f_mad = (1.4826 * np.nanmedian(np.abs(flat - med), axis=0)) ** 2
    return f_mean, f_mad
```

`run()`:

1. Load the empty dataset; compute both estimators per band; choose by `estimator`.
2. Also compute, per band, the lag-1 autocorrelation of the whitened pixels along rows and columns, and the mean of the whitened pixels (should be near 0; a non-zero mean signals a background offset).
3. Write `noise_model.yml`:

```yaml
bands: [u, g, r, i, z, y]
noise_factor: [..., ..., ..., ..., ..., ...]     # the f_b used downstream
noise_factor_mean: [...]
noise_factor_mad: [...]
lag1_corr: [...]
whitened_mean: [...]
n_stamps: 400
likelihood: measured_noise
```

4. QA: histogram of whitened pixels per band against a unit Gaussian; `chi2 / pixel` per stamp before and after `f_b`.

**Done when:** after dividing by `f_b`, the empty-sky `chi2 / pixel` is 1 within its statistical error in every band, and `f_b` lies roughly between 1.0 and 1.15 (section 2.1 found 1.08 by the plain mean in i). Much larger values mean the empty-sky selection is contaminated.

Keep the empty stamps: the L2 likelihood will later need `S_b,k` from the same set.

### 10.3 `CheckKernels` (train pipeline)

Uses the kernel code that will live in debvader-keras (section 11.3), so that the validation exercises exactly the function the loader uses.

`run()`:

1. Read `psf_stats.yml`; refuse to run if `ok` is false.
2. Build `G_int` per band with `debvader.physics.forward.internal_psf_stamp`.
3. For every object (or `max_objects` random ones) and band: `K, absK = matching_kernel(psf, g_int, lam)`; record `max |K~|` and the reconstruction error `max |K * G - P| / max P`.
4. Assert `max |K~| <= kernel_max` and reconstruction error `<= kernel_recon_tol` for all. On failure, write the report with the offending object ids and raise.
5. Write `kernel_report.yml` with the per-band maxima and the `internal_psf` block that was used.
6. QA: histograms of both quantities per band; a montage of `P`, `G`, `K` and `K * G - P` for the worst five objects.

If it fails:

- failure at low `k` (the `|K~|` map peaks near the centre): raise `beta` of the Moffat, or lower the FWHM;
- failure at high `k`: lower the FWHM, or raise `kernel_reg`;
- failure for a handful of objects only: inspect their PSF stamps; they may carry the `INEXACT_PSF` flag and be worth cutting.

**Done when:** the assertion passes for every object and band of the single-patch set.

---

## 11. Phase 5: debvader-keras, forward model and measured-noise likelihood

This is the core of the method. Do it on a branch, behind config switches, so the existing behaviour is untouched when the switches are off. No training is needed until section 11.9.

### 11.1 Summary of changes

| File | Action | Content |
|---|---|---|
| `src/debvader/utils/config.py` | modify | new fields (11.2) |
| `src/debvader/physics/__init__.py` | create | empty |
| `src/debvader/physics/forward.py` | create | `odd_pixels`, `internal_psf_stamp`, `matching_kernel`, `convolve_same` (numpy and keras), `ObservationOperator` (11.3) |
| `src/debvader/core/decoder.py` | modify | output channel count and activation from config (11.4) |
| `src/debvader/core/vae.py` | modify | `decoder_mean`, measured-noise loss, chi2 metric (11.5) |
| `src/debvader/data/augmentation.py` | modify | reusable `d4_params` / `apply_d4` (11.6) |
| `src/debvader/data/loader.py` | modify | `ForwardModelDataset`, `get_forward_model_dataset` (11.6) |
| `src/debvader/training/trainer.py` | modify | branch on `likelihood`; `steps_per_epoch`; channel helper (11.7) |
| `src/debvader/training/callbacks.py` | modify | use `decoder_mean` at lines 575, 1450, 1501 |
| `src/debvader/inference/predictor.py` | modify | use `decoder_mean`; add `predict_forward` (11.8) |
| `src/debvader/utils/paths.py` | modify | `set_experiments_dir` (11.7) |
| `src/debvader/api.py` | create | `train()`, `load_model()` with explicit paths (11.7) |
| `tests/test_forward_model.py` | create | the offline tests (11.9) |
| `tests/test_measured_noise_loss.py` | create | loss tests (11.9) |

### 11.2 Config fields

`utils/config.py`, additions (all defaults reproduce today's behaviour):

```python
@dataclass
class DataConfig:
    ...
    bands: List[str] = field(default_factory=lambda: ["u", "g", "r", "i", "z", "y"])
    pixel_scale: float = 0.2
    # f_b of the measured-noise likelihood, one per band. Empty = all ones.
    noise_factor_per_band: List[float] = field(default_factory=list)


@dataclass
class ModelConfig:
    ...
    # Number of image channels the decoder emits. 0 keeps the legacy head: 2*C channels,
    # mean and learned scale. Set to C for the measured-noise likelihood.
    n_out_channels: int = 0
    output_activation: str = "relu"      # "relu", "softplus" or "linear"


@dataclass
class TrainingConfig:
    ...
    likelihood: str = "learned_sigma"    # or "measured_noise"; "measured_noise_fourier" reserved
    forward_model: str = "none"          # or "kernel"
    internal_psf: dict = field(default_factory=dict)   # {type, beta, fwhm_arcsec}
    kernel_reg: float = 1e-3
```

Also make `Config.from_json` strict on request: add a `strict: bool = False` argument that raises on unknown keys. The pipeline never goes through JSON for input, but the snapshot written by `snapshot_run` is read back by the predictor.

### 11.3 `physics/forward.py` (new)

Numpy reference implementations (used by the loader, by `CheckKernels` and by the tests):

```python
import numpy as np


def odd_pixels(size_arcsec, pixel_scale):
    n = int(round(size_arcsec / pixel_scale))
    return n if n % 2 == 1 else n + 1


def internal_psf_stamp(size, fwhm_pix, kind="moffat", beta=4.765):
    """G_int sampled at pixel centres on a size x size grid, centred on size // 2, unit sum."""
    if size % 2 == 0:
        raise ValueError("PSF stamps must have an odd size")
    c = size // 2
    y, x = np.mgrid[:size, :size] - c
    r2 = x * x + y * y
    if kind == "moffat":
        alpha = fwhm_pix / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
        g = (1.0 + r2 / alpha**2) ** (-beta)
    elif kind == "gaussian":
        sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        g = np.exp(-0.5 * r2 / sigma**2)
    else:
        raise ValueError(kind)
    return g / g.sum()


def matching_kernel(psf, g_int, lam=1e-3):
    """K with K * G = P (regularised), for stamps centred on size // 2.

    psf   : (..., P, P), unit sum
    g_int : (P, P) or broadcastable, unit sum
    Returns (K, absK): K (..., P, P) with unit sum, and |K~| on the FFT grid.
    """
    ax = (-2, -1)
    Pk = np.fft.fft2(np.fft.ifftshift(psf, axes=ax), axes=ax)      # centre pixel -> index 0
    Gk = np.fft.fft2(np.fft.ifftshift(g_int, axes=ax), axes=ax)
    Kk = Pk * np.conj(Gk) / (np.abs(Gk) ** 2 + lam**2)
    K = np.fft.fftshift(np.fft.ifft2(Kk, axes=ax).real, axes=ax)   # index 0 -> centre pixel
    K = K / K.sum(axis=ax, keepdims=True)
    return K, np.abs(Kk)


def convolve_same_np(g, k):
    """Zero-padded linear convolution, 'same' size. g (..., H, W), k (..., P, P) centred."""
    H, W = g.shape[-2:]
    P = k.shape[-1]
    N, M = H + P - 1, W + P - 1
    full = np.fft.irfft2(np.fft.rfft2(g, s=(N, M)) * np.fft.rfft2(k, s=(N, M)), s=(N, M))
    c = P // 2
    return full[..., c:c + H, c:c + W]
```

Why these conventions:

- For an odd size `P`, `ifftshift` moves index `P // 2` to index 0, and `fftshift` moves it back. With even sizes the two differ and a half-pixel shift appears; this is why the odd rule matters.
- `convolve_same_np` pads to `H + P - 1`, so nothing wraps around the stamp edge. Model flux that would fall outside the stamp is simply lost, which is the correct behaviour for a stamp.
- `K` is computed on the `P x P` grid with periodic boundaries. `K` is compact (it is narrower than `P`), so wrap-around is negligible; the reconstruction check in `CheckKernels` measures it.

Backend-agnostic version used inside the loss. `keras.ops.fft2` takes and returns a `(real, imag)` tuple and transforms the last two axes (verified in Keras 3.15.1 with the JAX backend):

```python
import keras.ops as ops


def convolve_same(g, k):
    """g: (B, H, W, C). k: (B, P, P, C), centred on P // 2. Returns (B, H, W, C)."""
    H, W = g.shape[1], g.shape[2]
    P = k.shape[1]
    N, M = H + P - 1, W + P - 1
    g = ops.transpose(g, (0, 3, 1, 2))                          # (B, C, H, W)
    k = ops.transpose(k, (0, 3, 1, 2))
    g = ops.pad(g, [[0, 0], [0, 0], [0, N - H], [0, M - W]])
    k = ops.pad(k, [[0, 0], [0, 0], [0, N - P], [0, M - P]])
    gr, gi = ops.fft2((g, ops.zeros_like(g)))
    kr, ki = ops.fft2((k, ops.zeros_like(k)))
    mr = gr * kr - gi * ki                                      # complex product, real part
    mi = gr * ki + gi * kr
    m, _ = ops.ifft2((mr, mi))
    c = P // 2
    m = m[:, :, c:c + H, c:c + W]
    return ops.transpose(m, (0, 2, 3, 1))
```

The observation operator, as a small class so the joint model can replace it:

```python
class ObservationOperator:
    """Model frame -> observed frame. Test I: same grid, kernel convolution only."""

    def __init__(self, model_pixel_scale, obs_pixel_scale):
        if abs(model_pixel_scale - obs_pixel_scale) > 1e-9:
            raise NotImplementedError(
                "model and observed grids differ: resampling is the joint-model extension")
        self.model_pixel_scale = model_pixel_scale
        self.obs_pixel_scale = obs_pixel_scale

    def __call__(self, g, kernel):
        return convolve_same(g, kernel)
```

When the grids differ, the pixel response must be handled explicitly: a coadd PSF already contains the data pixel, and binning a finer model grid applies it again.

### 11.4 Decoder

`core/decoder.py`. Today:

```python
        self.final_conv: layers.Conv2D = layers.Conv2D(
            filters=input_shape[-1] * 2,
            kernel_size=3,
            padding="same",
            activation="relu",
        )
```

Change to:

```python
        n_out = int(getattr(config, "n_out_channels", 0)) or input_shape[-1] * 2
        act = getattr(config, "output_activation", "relu")
        self.final_conv: layers.Conv2D = layers.Conv2D(
            filters=n_out,
            kernel_size=3,
            padding="same",
            activation=None if act == "linear" else act,
        )
```

Notes:

- With `n_out_channels = 0` nothing changes for existing configs and weights.
- Keep `relu` as the Test I default: a non-negative galaxy is a reasonable prior and it is what the architecture was tuned with. If training stalls with many exactly-zero output pixels, switch to `softplus`.
- The crop logic already handles any stamp size: `start_dim = ceil(H / 16)`, then crop from `16 * start_dim` to `H`. The crop is asymmetric for odd differences (for 59: 2 px on one side, 3 on the other). This is harmless because the network is trained with that crop.
- `Dense(560)` is hard-coded and independent of the latent size. Leave it.

### 11.5 VAE loss

`core/vae.py`.

Add a helper and use it wherever the code slices `[..., :C]`:

```python
    def decoder_mean(self, dec_params):
        """The image channels of the decoder output, whatever the head layout."""
        if int(getattr(self.config, "n_out_channels", 0)) > 0:
            return dec_params
        c = ops.shape(dec_params)[-1] // 2
        return dec_params[..., :c]
```

In `__init__`:

```python
        self.likelihood: str = getattr(training_config, "likelihood", "learned_sigma")
        self.use_forward_model: bool = getattr(training_config, "forward_model", "none") == "kernel"
        if self.likelihood == "measured_noise_fourier":
            raise NotImplementedError("the Fourier (L2) likelihood is a reserved hook")
        self.chi2_tracker = keras.metrics.Mean(name="chi2_per_pixel")
```

and append `self.chi2_tracker` to `metrics` when `self.likelihood == "measured_noise"`. Build the `ObservationOperator` here when `use_forward_model` is true; it needs the two pixel scales, so pass them into the constructor (add `model_pixel_scale` / `obs_pixel_scale` keyword arguments defaulting to 0.2, or read them from a config field).

New loss method:

```python
    def _compute_recon_loss_measured(self, y, dec_params, sample_weight):
        """0.5 * chi2 / N_pix with measured noise.

        y is a dict:
          "image"  (B, H, W, C)  target in units of s_b
          "weight" (B, H, W, C)  s_b^2 / (f_b sigma^2), 0 on masked pixels
          "kernel" (B, P, P, C)  matching kernel K, present when the forward model is on
        """
        d = self.decoder_mean(dec_params)
        m = self.observe(d, y["kernel"]) if self.use_forward_model else d
        w = ops.cast(y["weight"], m.dtype)
        r = ops.cast(y["image"], m.dtype) - m
        chi2 = ops.sum(w * ops.square(r), axis=(1, 2, 3))                   # (B,)
        n_pix = ops.cast(ops.prod(ops.shape(m)[1:]), "float32")
        n_good = ops.maximum(ops.sum(ops.cast(w > 0, "float32"), axis=(1, 2, 3)), 1.0)
        self.chi2_tracker.update_state(chi2 / n_good)
        per_sample = 0.5 * chi2 / n_pix
        if sample_weight is None:
            return ops.mean(per_sample)
        sw = ops.cast(sample_weight, "float32")
        return ops.sum(per_sample * sw) / ops.maximum(ops.sum(sw), 1.0)
```

In `compute_loss`, branch at the top:

```python
        if self.likelihood == "measured_noise":
            if self.use_latent_head:
                raise NotImplementedError("heads + measured-noise likelihood: not wired yet")
            dec_params = y_pred
            recon_loss = self._compute_recon_loss_measured(y, dec_params, sample_weight)
            recon_weight = self.training_config.loss_weights.get("recon", 1.0)
            kl_losses = sum(self.losses)
            total_loss = recon_loss * recon_weight + kl_losses
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(recon_loss)
            self.kl_loss_tracker.update_state(kl_losses)
            return total_loss
```

and leave the rest of the method as it is.

Things to know about the KL term:

- `self.losses` holds `kl_weight * mean(KL)` added by the MVN layer. `kl_weight` is a `keras.Variable` driven by `KLAnnealingCallback` from `kl_start_value` to `kl_target_value` over `annealing_epochs`. `loss_weights["kl"]` only sets its initial value. Set **both** `loss_weights.kl` and `kl_target_value` to the same number (the config helper does).
- The logged `kl_loss` is therefore already multiplied by the weight. To read the raw KL in nats, divide by the current weight.

Also fix while you are here: `vae.py:312` (`_compute_shape_loss`) should use `self.decoder_mean`.

### 11.6 Augmentation and the dataset class

`data/augmentation.py`. Factor the geometric transform out so it can be applied to any array that must follow the image:

```python
def d4_params(rng=np.random):
    """One random element of the square's symmetry group: (flip_lr, flip_ud, n_quarter_turns)."""
    return bool(rng.random() > 0.5), bool(rng.random() > 0.5), int(rng.randint(4))


def apply_d4(a, lr, ud, k):
    """Apply the transform to an (H, W, C) array. Works for images, variance, masks and PSFs."""
    if lr:
        a = a[:, ::-1, :]
    if ud:
        a = a[::-1, :, :]
    if k:
        a = np.rot90(a, k, axes=(0, 1))
    return np.ascontiguousarray(a)
```

Rewrite `augment_data` in terms of these two so there is one implementation.

Why transforming the PSF is exact: flips and quarter turns map the pixel grid onto itself, and convolution commutes with them. A scene `x = P * g + n` transformed by `T` equals `T(P) * T(g) + T(n)`. `G_int` is circular, so `T(G) = G`, and the kernel of the transformed PSF is the transformed kernel. It is therefore enough to transform the PSF stamp and compute `K` afterwards.

`data/loader.py`. New dataset class:

```python
class ForwardModelDataset(keras.utils.PyDataset):
    """Batches for the measured-noise likelihood with a forward-modelled PSF.

    Yields (x, y) with
      x            (B, H, W, C)  encoder input, normalised (tanh-arcsinh)
      y["image"]   (B, H, W, C)  target / s_b
      y["weight"]  (B, H, W, C)  s_b^2 / (f_b * variance), 0 where masked
      y["kernel"]  (B, P, P, C)  matching kernel K from the PSF stamp and G_int
    Arrays are memory-mapped; nothing is normalised ahead of time.
    """

    def __init__(self, inputs, targets, variance, mask, psf, indices, *, scale, noise_factor,
                 g_int, kernel_reg, normalize_input, batch_size, augment, shuffle, seed=None,
                 workers=0, **kwargs):
        super().__init__(workers=max(workers, 1), use_multiprocessing=workers > 0, **kwargs)
        self.inputs, self.targets, self.variance, self.mask, self.psf = \
            inputs, targets, variance, mask, psf
        self.scale = np.asarray(scale, np.float32)                  # (C,)
        self.noise_factor = np.asarray(noise_factor, np.float32)    # (C,)
        self.g_int = np.asarray(g_int, np.float64)                  # (P, P, C)
        self.kernel_reg = float(kernel_reg)
        self.normalize_input = normalize_input                      # callable on (.., H, W, C)
        self.batch_size, self.augment, self.shuffle = int(batch_size), augment, shuffle
        self._rng = np.random.default_rng(seed)
        self._indices = np.array(indices)
        if shuffle:
            self._rng.shuffle(self._indices)

    def __len__(self):
        return len(self._indices) // self.batch_size if self.shuffle \
            else int(np.ceil(len(self._indices) / self.batch_size))

    def __getitem__(self, b):
        idx = np.sort(self._indices[b * self.batch_size:(b + 1) * self.batch_size])  # sorted: faster mmap
        xs, ts, ws, ks = [], [], [], []
        for i in idx:
            x = np.asarray(self.inputs[i], np.float32)
            t = np.asarray(self.targets[i], np.float32)
            v = np.asarray(self.variance[i], np.float32)
            m = np.asarray(self.mask[i], bool)
            p = np.asarray(self.psf[i], np.float64)
            if self.augment:
                lr, ud, k = d4_params(self._rng_legacy())
                x, t, v, m, p = (apply_d4(a, lr, ud, k) for a in (x, t, v, m, p))
            good = (~m) & np.isfinite(v) & (v > 0)
            w = np.where(good, self.scale**2 / (self.noise_factor * np.where(good, v, 1.0)), 0.0)
            K, _ = matching_kernel(np.moveaxis(p, -1, 0), np.moveaxis(self.g_int, -1, 0),
                                   self.kernel_reg)                 # (C, P, P)
            xs.append(self.normalize_input(np.where(np.isfinite(x), x, 0.0)))
            ts.append(np.where(good, t / self.scale, 0.0))
            ws.append(w)
            ks.append(np.moveaxis(K, 0, -1))
        y = {"image": np.stack(ts).astype(np.float32),
             "weight": np.stack(ws).astype(np.float32),
             "kernel": np.stack(ks).astype(np.float32)}
        return np.stack(xs).astype(np.float32), y

    def on_epoch_end(self):
        if self.shuffle:
            self._rng.shuffle(self._indices)
```

Implementation notes:

- `d4_params` as written takes a `numpy.random`-style object (`random()`, `randint()`). With a `Generator`, use `rng.random()` and `rng.integers(4)`; pick one style and make `d4_params` match it. The `self._rng_legacy()` placeholder above stands for whichever you choose.
- With `workers > 0`, each worker process gets a copy of the dataset object including its RNG state. Seed per batch (`np.random.default_rng([seed, epoch, b])`) if you need distinct augmentation streams across workers.
- `normalize_input` is `lambda a: normalize_nonlinear_numpy(a, scale, beta=2.5)` from `data/normalization.py`.
- Masked pixels in the encoder input: they are left as they are (after replacing non-finite values with 0). With the mask fraction kept low this is fine; do not zero them, since a hole in a galaxy is worse than a slightly bad pixel.
- The `(x, y)` tuple with a dict `y` is passed through `model.fit` to `compute_loss` unchanged.

Loader method:

```python
    def get_forward_model_dataset(self, stage, *, training_config, batch_size=None, num_workers=0):
        """Dataset for the measured-noise likelihood, read from a real-build directory."""
        from debvader.physics.forward import internal_psf_stamp
        d = self.real_dir
        if d is None:
            raise ValueError("the measured-noise likelihood needs a dataset directory")
        need = ("images.npy", "variance.npy", "mask.npy", "psf.npy", "meta.npz", "stats.json")
        missing = [f for f in need if not (d / f).is_file()]
        if missing:
            raise FileNotFoundError(f"{d} lacks {missing}")
        meta = np.load(d / "meta.npz")
        want = 1 if "val" in stage else (2 if "test" in stage else 0)
        idx = np.flatnonzero(np.asarray(meta["split"]) == want)
        images = np.load(d / "images.npy", mmap_mode="r")
        targets = np.load(d / "targets.npy", mmap_mode="r") if (d / "targets.npy").is_file() else images
        psf = np.load(d / "psf.npy", mmap_mode="r")
        scale = np.asarray(json.load(open(d / "stats.json"))["nonlinear"]["scale_per_band"], np.float32)
        C, P = images.shape[-1], psf.shape[1]
        f = np.asarray(self.config.noise_factor_per_band or np.ones(C), np.float32)
        ip = training_config.internal_psf
        fwhm = np.broadcast_to(np.atleast_1d(ip["fwhm_arcsec"]), (C,)) / self.config.pixel_scale
        g_int = np.stack([internal_psf_stamp(P, fw, ip.get("type", "moffat"), ip.get("beta", 4.765))
                          for fw in fwhm], axis=-1)
        is_train = "train" in stage
        return ForwardModelDataset(
            images, targets, np.load(d / "variance.npy", mmap_mode="r"),
            np.load(d / "mask.npy", mmap_mode="r"), psf, idx,
            scale=scale, noise_factor=f, g_int=g_int, kernel_reg=training_config.kernel_reg,
            normalize_input=lambda a: normalize_nonlinear_numpy(a, scale, beta=2.5),
            batch_size=batch_size or self.config.batch_size, augment=is_train, shuffle=is_train,
            seed=getattr(self.config, "seed", None), workers=num_workers)
```

`self.real_dir` is set in `DebvaderDataLoader.__init__` when `dataset_path` contains `images.npy` and `meta.npz`; that detection stays as it is.

### 11.7 Trainer, paths and an explicit-path API

**`utils/paths.py`.** The module reads `EXPERIMENTS_DIR` at import. Add a setter so a caller can direct outputs without environment variables:

```python
def set_experiments_dir(path) -> None:
    """Route every run directory under `path` (overrides the EXPERIMENTS_DIR environment variable)."""
    global EXPERIMENTS_DIR
    EXPERIMENTS_DIR = Path(os.path.expandvars(str(path))).expanduser()
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
```

`get_run_dir()` reads the module global at call time, so this is enough.

**`training/trainer.py`.** Three changes.

1. In `__init__`, the model warm-up and normaliser set-up stay. The normaliser is still built from `stats.json` through `resolve_stats_path(config.data.dataset_path)`, which already insists that a real build carries its own `stats.json`.

2. In `train_stage1` and `train_stage2`, branch on the likelihood before the data is loaded:

```python
        measured = getattr(self.config.training, "likelihood", "learned_sigma") == "measured_noise"
        if measured:
            train_ds = self.loader.get_forward_model_dataset(
                "train_clean", training_config=self.config.training)
            val_ds = self.loader.get_forward_model_dataset(
                "val_clean", training_config=self.config.training)
            # a small in-memory sample of normalised inputs for the monitoring callbacks
            val_images = np.concatenate([val_ds[i][0] for i in range(min(4, len(val_ds)))])
            callbacks = self.get_callbacks("stage1", val_images=val_images)
        else:
            ... existing code ...
```

   Use `"train_blended"` / `"val_blended"` in stage 2. Check what `get_callbacks` does when `train_images` and the metadata are `None`; the monitoring block is wrapped in a `try/except` that logs and continues, but make sure the checkpoint callbacks (which are outside it) do not need them.

3. `steps_per_epoch`. Today: `self.config.data.train_size // self.config.data.batch_size`, which is 0 when `train_size` is 0 (as in `configs/real_stage1.json`). Pass `None` when the result is 0 so that Keras uses `len(train_ds)`:

```python
        steps_per_epoch = (self.config.data.train_size // self.config.data.batch_size) or None
```

   Find out how the existing real-data runs behave with 0 before changing it for the legacy path.

Stage 2 specifics that already exist and that the pipeline relies on:

- it requires `stage1_best.weights.h5` in the run's `checkpoints/` directory;
- it sets `self.model.decoder.trainable = False` and recompiles;
- the encoder starts from the Stage 1 weights because the whole model is loaded first.

**`api.py` (new).** A thin, importable version of `scripts/train.py`, with explicit paths:

```python
def train(config, stage, experiments_dir, run_name, resume=True, init_weights=None, seed=None):
    """Train stage 1 or 2 with every output under experiments_dir/run_name.

    init_weights: for stage 2, the stage-1 weights file to start from. It is copied to
    <run>/checkpoints/stage1_best.weights.h5, which is where train_stage2 looks.
    Returns the path of the best weights of the stage.
    """
    import shutil
    from debvader.utils import paths
    from debvader.utils.logging import capture_run_logs, setup_logger
    from debvader.utils.seeding import set_global_seed

    paths.set_experiments_dir(experiments_dir)
    paths.set_run_name(run_name)
    paths.ensure_workspace_exists()
    logger = setup_logger()
    capture_run_logs(logger)
    set_global_seed(config.data.seed if seed is None else seed)
    paths.snapshot_run(config)
    if stage == 2 and init_weights is not None:
        shutil.copy(init_weights, paths.get_checkpoints_dir() / "stage1_best.weights.h5")

    from debvader.training.trainer import DebvaderTrainer
    trainer = DebvaderTrainer(config, resume=resume)
    (trainer.train_stage1 if stage == 1 else trainer.train_stage2)()
    return paths.get_checkpoints_dir() / f"stage{stage}_best.weights.h5"
```

Check the actual file name of the best checkpoint in `get_callbacks` (the Stage 2 loader looks for `stage1_best.weights.h5`, so Stage 1 writes that; confirm the Stage 2 name).

`scripts/train.py` also calls `configure_distribution` for multi-GPU. Leave multi-GPU out of `api.train` at first (single device), and add it once single-GPU training works.

### 11.8 Predictor

`inference/predictor.py` currently normalises the input, calls the model, slices the mean channels and applies the inverse normaliser. Under the new scheme the inverse of the output is a multiplication by `s_b`, not the inverse of `tanh(arcsinh)`.

Add a method (and replace the two `params[..., :channels]` slices with `self.model.decoder_mean(params)`):

```python
    def predict_forward(self, images, psf, variance=None, mask=None, batch_size=512):
        """Deterministic prediction for the forward-model scheme.

        Returns a dict of numpy arrays:
          g        (N, H, W, C)  model at G_int, flux units
          m        (N, H, W, C)  model at the observed PSF, flux units
          z_mean   (N, D)
          z_sigma  (N, D)        sqrt(diag(L L^T))
          chi2_pix (N,)          only when variance is given
        """
```

Steps inside, per batch:

1. `u = normalize_nonlinear_numpy(images, scale)`.
2. `enc = self.model.encoder(u, training=False)`; `mu = enc[:, :D]`.
3. Build `L` the same way the MVN layer does (lower-triangular fill, `softplus(diag) + 1e-5`). Factor that construction out of `KerasMultivariateNormalTriL.call` into a method `cholesky(params)` so the predictor and the layer share it. `z_sigma = sqrt(sum(L**2, axis=-1))`.
4. `d = self.model.decoder_mean(self.model.decoder(mu, training=False))`; `g = d * scale`.
5. `K = matching_kernel(psf, g_int, lam)`; `m = convolve_same_np(g, K)` (numpy, channel-first internally).
6. If `variance` is given: `w = (~mask) / (f * variance)`; `chi2_pix = sum(w * (images - m)**2) / sum(w > 0)`.

`BatchNormalization` in the encoder uses its moving statistics when `training=False`; that is what you want at inference.

The model directory must carry everything needed to do this without the training dataset: `weights.h5`, `config.json`, and `model_card.json` with `scale_per_band`, `noise_factor`, `internal_psf`, `kernel_reg`, `bands`, `pixel_scale`, `stamp_arcsec`, `stamp_px`, `psf_px`.

### 11.9 Offline tests, before any real training

`tests/test_forward_model.py`. No network involved.

**Test 1: the kernel reproduces the PSF.**

```python
def test_kernel_reproduces_psf():
    P = 35
    g = internal_psf_stamp(P, 2.5, "moffat", 4.765)
    psf = internal_psf_stamp(P, 5.0, "moffat", 3.5)          # a wider, different profile
    K, absK = matching_kernel(psf, g, lam=1e-3)
    recon = convolve_same_np(g, K)
    assert np.abs(recon - psf).max() / psf.max() < 1e-3
    assert absK.max() < 1.02
    assert abs(K.sum() - 1) < 1e-6
```

Repeat with real PSF stamps from the exported dataset, and with an elliptical, slightly off-centre PSF.

**Test 2: the forward model against GalSim.** This is the test that catches half-pixel shifts and normalisation errors together.

```python
def test_forward_model_matches_galsim():
    import galsim
    n, P, scale = 59, 35, 0.2
    gal = galsim.Sersic(n=1.5, half_light_radius=0.6, flux=1000.0).shear(e1=0.2, e2=-0.1)
    g_prof = galsim.Moffat(beta=4.765, fwhm=0.5)
    p_prof = galsim.Moffat(beta=3.5, fwhm=1.0).shear(e1=0.03, e2=0.02)
    off = (0.3, -0.2)                                             # sub-pixel offset, in pixels

    def draw(prof, size, offset=(0, 0)):
        return prof.drawImage(nx=size, ny=size, scale=scale, offset=offset, method="no_pixel").array

    g_stamp = draw(g_prof, P); g_stamp /= g_stamp.sum()
    p_stamp = draw(p_prof, P); p_stamp /= p_stamp.sum()
    at_g = draw(galsim.Convolve(gal, g_prof), n, off)             # what the decoder should emit
    at_p = draw(galsim.Convolve(gal, p_prof), n, off)             # what the data would show
    K, _ = matching_kernel(p_stamp, g_stamp, lam=1e-3)
    model = convolve_same_np(at_g, K)
    assert np.abs(model - at_p).max() / at_p.max() < 5e-3
```

Notes:

- `method="no_pixel"` is used for every draw because the PSF stamps are treated as effective PSFs (pixel included), as the coadd PSF is. `poc_injection.py` documents the same choice.
- GalSim's default centre for an odd-sized image is the centre pixel, matching the convention here.
- The tolerance is looser than Test 1 because the profiles are truncated by the stamps.
- Also assert that `convolve_same` (keras, float32) equals `convolve_same_np` to a relative 1e-5 on random inputs, and that `convolve_same_np` equals `scipy.signal.fftconvolve(..., mode="same")` to 1e-10.
- Reference values from running exactly the code in this guide on 2026-11-12: Test 1 error 6e-5, Test 2 error 1.4e-3, augmentation test (Test 4) 2e-7.

**Test 3: empty sky gives chi-squared per pixel of 1.** Take the empty dataset, build `ForwardModelDataset` batches, set the model to zero, and evaluate `sum(w * image**2) / sum(w > 0)` on the normalised quantities. It must equal 1 within the statistical error, per band. This validates `sigma`, `f_b`, `s_b` and the mask in the units the loss actually sees.

**Test 4: augmentation consistency.** For a synthetic stamp `x = convolve(g, P)`, check that for all eight transforms `T`, `convolve(T(g), K(T(P)))` equals `T(x)` to 1e-6.

`tests/test_measured_noise_loss.py`:

- Build a tiny `DebvaderVAE` with `n_out_channels = C`, `likelihood = "measured_noise"`. Check that `call()` returns `(B, H, W, C)`.
- With a hand-made `y` and `y_pred`, check `compute_loss` against a numpy evaluation of `0.5 * chi2 / N_pix + kl`.
- Check that with all switches off the legacy tests still pass (`pytest tests/`).

**Overfit a tiny batch.** The single most valuable check.

1. Take 16 isolated galaxies (bright ones, from the single-patch dataset). Make a dataset directory containing just those, with `split = 0` for all, and use the same 16 for validation.
2. Config: `kl` weight 0 (pure reconstruction), `learning_rate` 1e-3, no augmentation, batch size 16, 2000 epochs.
3. Train on CPU or one GPU; it takes minutes.
4. Expect: `chi2_per_pixel` falls toward and slightly below 1 (the network is fitting noise in 16 images, which is expected here), and residual images `(x - K * g) / sigma` show no galaxy-shaped structure.

If the residuals keep a galaxy-shaped pattern, or `chi2_per_pixel` plateaus far above 1, the bug is in the code. Typical culprits, in order: normalisation units, PSF centring, `K` applied to the wrong band, augmentation not applied to the PSF, dead ReLU outputs.

**Done when:** Tests 1 to 4 pass, the loss tests pass, the legacy test suite still passes, and the 16-galaxy batch overfits.

---

## 12. Phase 6: `TrainPrior`

### 12.1 The stage

```python
# src/dbv_pipeline/train.py
class TrainPrior(PipelineStage):
    name = "TrainPrior"
    inputs = [("isolated_cutouts", DebvaderDataset), ("noise_model", YamlFile),
              ("kernel_report", YamlFile)]
    outputs = [("prior_model", ModelDirectory), ("qa_prior", Directory)]
    config_options = {"run_name": "prior", "work_dir": str, "debvader": dict,
                      "bands": [str], "pixel_scale": float, "stamp_arcsec": float,
                      "internal_psf": dict, "kernel_reg": 1e-3, "resume_training": True}

    def run(self):
        import json, shutil
        from pathlib import Path
        import yaml
        from .utils.config import build_debvader_config
        from .utils.geometry import stamp_geometry

        dataset = Path(self.get_input("isolated_cutouts"))
        noise = yaml.safe_load(open(self.get_input("noise_model")))
        report = yaml.safe_load(open(self.get_input("kernel_report")))
        self._check_consistency(dataset, noise, report)        # bands, pixel scale, G_int agree

        cfg = build_debvader_config(self.config, dataset, noise["noise_factor"], stage=1)
        work = Path(self.config["work_dir"]) / "training"       # NOT a ceci output: survives reruns

        from debvader import api
        best = api.train(cfg, stage=1, experiments_dir=work, run_name=self.config["run_name"],
                         resume=self.config["resume_training"])

        out = Path(self.get_output("prior_model"))
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, out / "weights.h5")
        cfg.save_json(out / "config.json")
        card = dict(stage=1, bands=list(self.config["bands"]),
                    pixel_scale=self.config["pixel_scale"],
                    stamp_arcsec=self.config["stamp_arcsec"], **stamp_geometry(self.config),
                    psf_px=json.load(open(dataset / "dataset.json"))["psf_px"],
                    internal_psf=dict(self.config["internal_psf"]),
                    kernel_reg=self.config["kernel_reg"],
                    noise_factor=noise["noise_factor"],
                    scale_per_band=json.load(open(dataset / "stats.json"))["nonlinear"]["scale_per_band"],
                    dataset=str(dataset))
        json.dump(card, open(out / "model_card.json", "w"), indent=2)
        self._write_qa(work / self.config["run_name"], dataset, out)
```

`_check_consistency`: compare `dataset.json` (bands, pixel scale, stamp size) and `kernel_report.yml` (`internal_psf`) with the global config, and raise on any difference. This is the guard against training with a `G_int` that was not the one validated.

How training interacts with ceci:

- ceci sees only `prior_model` and `qa_prior`. They are written at the very end, so an interrupted job leaves no outputs and ceci will run the stage again.
- The second run finds the checkpoints in `work_dir/training/prior/checkpoints` and debvader's own `--resume` logic continues from the last epoch. This is why the working directory must not be a declared output.
- debvader refuses to resume when the config fingerprint changed. If you change a hyperparameter, use a new `run_name` (or clear the work directory).
- To extend a finished run, raise `epochs` and delete `prior_model`; the epoch count is not part of the fingerprint.

### 12.2 Running it

First by hand on an interactive GPU node (the account for GPU hours is an open item):

```bash
salloc -N 1 -C gpu -G 1 -q interactive -t 02:00:00 -A <gpu_account>
conda activate debvader
python3 -m dbv_pipeline TrainPrior --config=pipelines/config_test1.yml \
    --isolated_cutouts=... --noise_model=... --kernel_report=... \
    --prior_model=... --qa_prior=...
```

Then through a batch script that runs `ceci pipelines/train.yml` inside the allocation with `site: local`. Only after that works, look at ceci's NERSC sites (`ceci/sites/nersc.py`) if you want ceci to submit jobs itself.

### 12.3 What to monitor

| Quantity | Where | Expectation |
|---|---|---|
| `val_chi2_per_pixel` | Keras logs | falls and plateaus near 1 |
| `val_kl_loss / kl_weight` | Keras logs | raw KL in nats; grows as the latent is used |
| active latent dimensions | `LatentActivityMonitorCallback` output under `logs/monitoring/stage1/latent_activity` | a stable number, well above 1; all 32 near zero means posterior collapse |
| residual montage | `qa_prior` | `(x - K * g) / (sqrt(f) sigma)` looks like noise |
| chi2 per pixel against magnitude | `qa_prior` | flat; a rise at the bright end means the decoder lacks capacity or `s_b` scaling is poor |

`_write_qa` should produce, from the validation split and the final weights (use `predict_forward`): the training curves, a montage of data / model at observed PSF / model at `G_int` / whitened residual for 16 random and the 16 worst objects, and the chi-squared distribution.

Tuning order:

1. `kl` weight: start at `1/N_pix`. If reconstructions are blurry and KL is tiny, anneal more slowly (`annealing_epochs`) before lowering the weight.
2. Learning rate: 1e-4 as in the existing real-data config.
3. Only then architecture (latent size, filters).

**Done when:** validation chi-squared per pixel plateaus near 1 on held-out patches and the residual montage shows no galaxy-shaped structure.

---

## 13. Phase 7: blends and `TrainDeblender`

### 13.1 Changes to `scripts/real_data/build_blends.py`

First make it importable: move the body of `main()` into a function in a new module `src/debvader/data/blends.py`, `build_blends(data_dir, out_dir, *, n_scenes, central, splits, empty_dir=None, dmag, theta, ..., seed)`, and keep the script as a thin argparse wrapper around it.

What it does today (summary): picks `n_scenes` targets from stamps with valid magnitude and shape; for each, draws 0 to 3 contaminants with uniform `dmag` and `theta`, shape/size/colour from a real template of similar magnitude in the same split, rendered with GalSim through the target's PSF (`method="no_pixel"`), plus shot noise `N(0, b_band * flux)`; writes `images.npy`, `targets.npy`, `psf.npy`, `meta.npz`.

Additions:

1. **Carry variance and mask.** Open `variance.npy` and `mask.npy` of the source; write for each scene

```
variance_blend = variance_target + b_band * clip(added, 0, None)
mask_blend     = mask_target
```

   where `added` is the noiseless contaminant light (compute the variance before the shot noise is added to `added`). Write `variance.npy` and `mask.npy` in the output.

2. **`splits` argument.** Restrict the target pool to the given splits (`[0, 1]` for the training set, `[2]` for evaluation sets) and keep the `split` value of the target in the output. Templates are already drawn within the target's split.

3. **`central="injected"` mode.** Needs `empty_dir`.

```
for each scene:
    pick an empty-sky stamp e (image, variance, mask, psf) from the requested split
    pick a template galaxy t (as for contaminants) and a magnitude from the target distribution
    central = Convolve(Sersic profile of t, psf_e) drawn noiselessly, with a random
              sub-pixel offset uniform in [-0.5, 0.5] px
    contaminants: exactly as today, drawn through psf_e
    scene    = e.image + central + contaminants + shot noise of (central + contaminants)
    target   = central                                  (noiseless)
    variance = e.variance + b_band * clip(central + contaminants, 0, None)
    psf      = psf_e
```

   Record the truth in `meta.npz`: per-band flux of the central (`truth_flux`, shape `(N, C)`), its half-light radius, ellipticity, offset, and the same contaminant columns as today.

4. **Write `dataset.json` and copy `stats.json`** from the source dataset into the output. The scale constants must be those of the isolated training set, never recomputed on blends.

5. **Record `truth_flux` in `central="real"` mode too** as NaN, so the two sets have the same columns.

Check `DEFAULT_B` (the per-band slope of the variance law) against your patches by running `measure_variance_law.py` on a few Butler coadds; it was measured on different patches. Adapt that script's `read_planes` to accept arrays so you can feed it `CellCoadd` planes.

### 13.2 `BuildBlends` stage

```python
class BuildBlends(PipelineStage):
    name = "BuildBlends"
    inputs = [("isolated_cutouts", DebvaderDataset), ("empty_cutouts", DebvaderDataset)]
    outputs = [("blend_cutouts", BlendDataset), ("eval_blends_real", BlendDataset),
               ("eval_blends_injected", BlendDataset)]
    config_options = {"n_train_scenes": 100000, "n_eval_scenes": 10000,
                      "dmag": [-1.0, 3.0], "theta": [0.325, 3.0], "seed": 0}

    def run(self):
        from debvader.data.blends import build_blends
        iso, empty = self.get_input("isolated_cutouts"), self.get_input("empty_cutouts")
        common = dict(dmag=self.config["dmag"], theta=self.config["theta"])
        build_blends(iso, self.get_output("blend_cutouts"), n_scenes=self.config["n_train_scenes"],
                     central="real", splits=[0, 1], seed=self.config["seed"], **common)
        build_blends(iso, self.get_output("eval_blends_real"), n_scenes=self.config["n_eval_scenes"],
                     central="real", splits=[2], seed=self.config["seed"] + 1, **common)
        build_blends(iso, self.get_output("eval_blends_injected"), empty_dir=empty,
                     n_scenes=self.config["n_eval_scenes"], central="injected", splits=[2],
                     seed=self.config["seed"] + 2, **common)
```

QA to write alongside (a `qa` sub-directory inside each output is simplest): a montage of blend / target / difference; histograms of `dmag`, `theta`, multiplicity; the fraction of contaminants brighter than the target.

Sanity checks to assert:

- `images - targets` in `central="real"` mode equals the injected light plus its shot noise, and is zero for scenes with `n_gal = 1`;
- `variance` of the blend is >= the variance of the target everywhere;
- every scene of an evaluation set has `split == 2`.

100 000 scenes at 59 x 59 x 6 take about 8 GB per array (images, targets, variance), so about 25 GB per training set. `merge_blends.py` exists if you want to build shards in parallel.

### 13.3 `TrainDeblender` stage

Same structure as `TrainPrior`, with:

```python
        prior = Path(self.get_input("prior_model"))
        prior_card = json.load(open(prior / "model_card.json"))
        self._check_same_frame(prior_card)     # bands, stamp size, G_int, scale_per_band, kernel_reg
        cfg = build_debvader_config(self.config, dataset, noise["noise_factor"], stage=2)
        best = api.train(cfg, stage=2, experiments_dir=work, run_name=self.config["run_name"],
                         resume=self.config["resume_training"], init_weights=prior / "weights.h5")
```

`_check_same_frame` must also compare `scale_per_band` in the blend dataset's `stats.json` with the prior's card. A mismatch means the frozen decoder would be read in the wrong units.

The architecture block under `TrainDeblender.debvader.model` must equal the one under `TrainPrior`. Rather than repeat it, read `model` from the prior's `config.json` and reject a `model` block in the `TrainDeblender` section.

What happens inside `train_stage2` (already implemented): the full Stage 1 model is loaded, the decoder is frozen, the optimiser is rebuilt, and the encoder trains on `(blend -> target)` pairs. With the measured-noise branch of section 11.7, the dataset reads `images.npy` as input, `targets.npy` as target, and `variance.npy` of the blend for the weights.

Monitoring:

- `val_chi2_per_pixel`: in `central="real"` mode the target is noisy, so the floor is near 1, as in Stage 1.
- KL: compare with Stage 1. A much larger KL means the blend encoder is pushing latents away from the prior; consider keeping `kl` at the Stage 1 value rather than lowering it.
- Montage: blend, target, `K * g`, and `(target - K * g) / sigma`.

**Done when:** validation chi-squared per pixel on blends is close to the Stage 1 value for scenes with `n_gal = 1` (they are identical to Stage 1 inputs), and degrades smoothly, not abruptly, with decreasing separation and with contaminants brighter than the target.

---

## 14. Phase 8: `RunDeblender` and `Evaluate`

### 14.1 `RunDeblender`

```python
class RunDeblender(PipelineStage):
    name = "RunDeblender"
    inputs = [("deblender_model", ModelDirectory), ("noise_model", YamlFile),
              ("eval_blends_real", BlendDataset), ("eval_blends_injected", BlendDataset)]
    outputs = [("deblended_outputs", HDFFile)]
    config_options = {"batch_size": 512, "eval_sets": ["eval_blends_real", "eval_blends_injected"]}
```

`run()` loads the model once (`DebvaderPredictor` from `config.json` + `weights.h5` + the card), then for each evaluation set calls `predict_forward(images, psf, variance, mask)` in chunks and writes one HDF5 group per set:

```
/eval_blends_injected/
    g          (N, H, W, C) float32   model at G_int, nJy
    m          (N, H, W, C) float32   model at observed PSF, nJy
    z_mean     (N, D)
    z_sigma    (N, D)
    chi2_pix   (N,)                   blend minus model: large when neighbours remain, by design
    chi2_pix_target (N,)              target minus model, weights from the blend variance
    index      (N,)                   row in the evaluation set
  attrs: bands, pixel_scale, internal_psf, model path
```

Use `self.open_output("deblended_outputs")` to get the `h5py.File` (provenance is written for you).

Interpretation of the two chi-squared values:

- `chi2_pix_target` measures deblending quality where truth exists. It is the headline number.
- `chi2_pix` against the **blend** is large whenever contaminants are present, since the model contains only the central object. It becomes the failure flag of the brief only after all members are modelled and summed (section 14.4). For Test I, compute it inside a central aperture (radius about 2 PSF FWHM) to make it usable as a flag.

Ten thousand scenes run in seconds on a GPU and in a few minutes on CPU; record the time per object for the comparison with scarlet.

### 14.2 `Evaluate`

Inputs: the HDF5 file and the two evaluation sets (for truth and metadata). Outputs: `metrics.yml` and figures.

Measured quantities per scene (write a small module `dbv_pipeline/metrics.py`, pure numpy plus galsim HSM):

| Quantity | Definition |
|---|---|
| model flux per band | `F_b = sum_i m_b,i` (model at the observed PSF) |
| true flux per band | injected set: `truth_flux`; real set: `sum_i target_b,i` (noisy) |
| magnitude residual | `dmag_b = -2.5 * log10(F_b / F_true_b)` |
| colour residual | `d(g-r) = dmag_g - dmag_r`, and likewise `r-i`, `i-z` |
| size and shape | HSM adaptive moments of `m` and of the target, same band, same weight; residuals in `e1`, `e2` and `sigma` |
| blendedness | from the scene: `1 - sum(target * target) / sum(target * blend)` in the i band (the Rubin definition, using the noiseless target where available) |
| baseline "no deblending" | the same measurements made directly on the blend |

Report bias (median) and scatter (normalised MAD) of each residual in bins of: number of galaxies, `dmag`, `theta_measured`, blendedness, target S/N, PSF FWHM, and sub-pixel offset. Give the injected and real sets separately, and scenes with contaminants brighter than the target (`dmag < 0`) separately from the rest.

Calibration of the latent uncertainty (injected set): encode the noiseless target alone to get `z_true`, then check that `(z_mean - z_true) / z_sigma` has unit scatter. This tests whether `z_sigma` tracks the actual error.

Figures: residual versus each binning variable with the no-deblending baseline overlaid; chi-squared distributions; a montage of the 16 best, 16 median and 16 worst scenes.

`metrics.yml`: the binned numbers, global summaries, counts, the fraction of scenes above `chi2_flag`, and the model and dataset identifiers.

### 14.3 Flagging unrecognised blends (real objects)

This uses no injection.

1. In DP2-cutouts, run `stamps` with `stamps.targets: all` on the held-out test patches, and export with `flux_raw` (no companion masking) to a dataset with `kind: "real_objects"`. Add `sb_euclid`, `has_euclid_counterpart` and the number of Euclid sources in the group to `meta.npz`.
2. Add it as a third evaluation set of `RunDeblender` (`targets.npy` absent, so only `g`, `m`, `z` and `chi2_pix` are written).
3. In `Evaluate`, report:
   - the fraction of Rubin objects whose group contains more than one Euclid source (the unrecognised-blend fraction), versus magnitude;
   - the distributions of `chi2_pix` (central aperture) and of the model flux relative to the Rubin cModel flux, for objects flagged as unrecognised blends against those that are one-to-one.

A difference between the two populations is the first evidence about whether the model responds to blends Rubin did not recognise.

### 14.4 Deferred: scene-level redistribution and the scarlet baseline

Not required to close Test I; listed so the design leaves room.

- **Redistribution** (`child_i = y * m_i / sum_j m_j`) needs every member of a blend run as the central object and all models placed on one scene grid. That requires scene-level cutouts (larger than a stamp) and stamp-to-scene bookkeeping (`offset_x`, `offset_y` and the integer stamp origin). Plan it together with the Euclid-centred cutouts of the joint test.
- **Scarlet baseline**: `object_scarlet_models` exists in the Butler for real objects. For injected scenes there is no Rubin deblender output, so the comparison must be made on real blends, with Euclid or HST as the reference.

**Done when:** `eval.yml` runs end to end, the injected-set flux and colour residuals are reported against blendedness and separation with the baseline alongside, and the unrecognised-blend fraction is reported.

---

## 15. Phase 9: scale to the full ECDFS field

Only after everything above works on one patch.

1. **Find the tracts and patches.**

```python
from lsst.daf.butler import Butler
butler = Butler("dp2", collections="dp2")
refs = butler.query_datasets(
    "deep_coadd",
    where="skymap = 'lsst_cells_v2' AND band = 'i' AND tract IN (5062, 5063, 5064)",  # adjust
    limit=None)
patches = sorted({(r.dataId["tract"], r.dataId["patch"]) for r in refs})
```

   Use the skymap (`butler.get("skyMap", skymap="lsst_cells_v2", collections="skymaps")`, then `findTractPatchList` on the corners of your sky box) to list candidate tracts, and keep only patches where all six bands exist.

2. **Sky box.** Enlarge `sky_box` to the overlap of the DP2 ECDFS coverage and the Euclid Q1 EDF-F footprint. Plot both catalogues before choosing the box.

3. **Catalogue steps.** `euclid_catalog` downloads by area; `copaings` cost grows with the number of objects. Go in steps (one patch, one tract, the field), timing each. Raise `n_eot_tasks` and `n_sb_tasks`. If copaings does not scale to the whole box at once, run it per tract with a margin and concatenate; the selection is local, so this is safe as long as the margin exceeds the linking length (3 arcsec).

4. **Stamps.** The per-patch shards make this restartable. Run on a compute node. If it is slow, parallelise over patches: ceci can run a stage under MPI (`nprocess: N`; inside `run()`, use `self.split_tasks_by_rank(patches)` to give each rank its share), with the export done by rank 0 only.

5. **Splits.** With many patches, remove the single-patch override, and check the realised train / validation / test counts. Hold the test patches out of everything, including `stats.json`, `psf_stats` percentiles used to choose `G_int`, and the blend template pool for training sets.

6. **Re-validate.** Re-run `MeasurePsf`, `MeasureNoise` and `CheckKernels` on the full set. If `MeasurePsf` now fails because a patch has a narrower PSF than `G_int`, you must lower `G_int` and retrain. This is why the full-field PSF histogram should be made **before** the long training runs: run the data pipeline on the full field first, then freeze `G_int`.

7. **Sample size.** Record the number of isolated training stamps. As a reference, the original debvader work used about 100 000 training stamps; if the count is far below that, watch the gap between training and validation chi-squared for overfitting.

---

## 16. Master checklist of files

### dbv-pipeline

| File | Action | Phase |
|---|---|---|
| `pyproject.toml`, `requirements.txt` | fix `setuptools_scm`, dependencies | 0 |
| `example/ceci_dummy_stage.yml` | `modules: dbv_pipeline` | 0 |
| `main.yaml`, `config.yaml` | delete | 0 |
| `src/dbv_pipeline/__init__.py` | version fallback, import stage modules | 0 |
| `src/dbv_pipeline/utils/provenance.py` | remove `distutils` | 0 |
| `src/dbv_pipeline/data_types/base.py`, `__init__.py` | bug fixes; `DebvaderDataset`, `BlendDataset`, `ModelDirectory`; `JSONFile` read/write | 0 |
| `src/dbv_pipeline/utils/geometry.py` | `odd_pixels`, `stamp_geometry` | 0 |
| `src/dbv_pipeline/utils/config.py` | `build_debvader_config`, `build_dp2_cutouts_config` | 0 |
| `pipelines/config_test1.yml`, `data_dp2.yml`, `train.yml`, `eval.yml` | create | 1 |
| `src/dbv_pipeline/select.py` | `SelectSources` | 1 stub, 3 real |
| `src/dbv_pipeline/cutouts.py` | `MakeCutouts` | 1 stub, 3 real |
| `src/dbv_pipeline/psf.py` | `MeasurePsf`, `CheckKernels` | 1 stub, 4 real |
| `src/dbv_pipeline/noise.py` | `MeasureNoise` | 1 stub, 4 real |
| `src/dbv_pipeline/train.py` | `TrainPrior`, `TrainDeblender` | 1 stub, 6 and 7 real |
| `src/dbv_pipeline/blends.py` | `BuildBlends` | 1 stub, 7 real |
| `src/dbv_pipeline/deblend.py` | `RunDeblender` | 1 stub, 8 real |
| `src/dbv_pipeline/evaluate.py`, `metrics.py` | `Evaluate` | 1 stub, 8 real |
| `src/dbv_pipeline/qa.py` | montage, histogram, residual-plot helpers | 3 onward |
| `tests/test_config_helpers.py`, `tests/test_pipeline_stubs.py` | create | 0, 1 |

### DP2-cutouts

| File | Action | Phase |
|---|---|---|
| `src/dp2_cutouts/sources/rubin_butler.py` | create: `fetch_objects`, `run_catalog`, `load_patch`, `psf_stamp` | 2 |
| `src/dp2_cutouts/common/psf.py` | create: `fwhm_from_stamp` | 2 |
| `src/dp2_cutouts/stages/stamps.py` | provider dispatch, patch loop and shards, PSF and variance in the arrays, bands / pixel scale / sizes from config, `targets: all`, `split_of`, offsets | 2 |
| `src/dp2_cutouts/stages/empty_sky.py` | create | 2 |
| `src/dp2_cutouts/stages/export_debvader.py` | create: `export_isolated`, `export_empty`, `write_stats`, `write_dataset_json` | 3 |
| `src/dp2_cutouts/cli.py` | new steps; dispatch on source | 2, 3 |
| `src/dp2_cutouts/sources/rubin_soda.py`, `stages/package.py`, `stages/catalog_table.py` | remove hard-coded `0.2` and `BANDS` | 2 |
| `config/butler_demo.yaml` | create: the demo box with `source: butler` | 2 |
| `tests/test_butler_source.py`, `tests/test_export.py` | create | 2, 3 |

### debvader-keras

| File | Action | Phase |
|---|---|---|
| `src/debvader/utils/config.py` | new fields; optional strict `from_json` | 5 |
| `src/debvader/physics/forward.py` | create | 5 |
| `src/debvader/core/decoder.py` | `n_out_channels`, `output_activation` | 5 |
| `src/debvader/core/vae.py` | `decoder_mean`, measured-noise loss, chi2 metric | 5 |
| `src/debvader/modules/layers.py` | factor out `cholesky(params)` | 5 |
| `src/debvader/data/augmentation.py` | `d4_params`, `apply_d4` | 5 |
| `src/debvader/data/loader.py` | `ForwardModelDataset`, `get_forward_model_dataset` | 5 |
| `src/debvader/training/trainer.py` | likelihood branch, `steps_per_epoch` | 5 |
| `src/debvader/training/callbacks.py` | `decoder_mean` at three places | 5 |
| `src/debvader/inference/predictor.py` | `decoder_mean`, `predict_forward` | 5, 8 |
| `src/debvader/utils/paths.py` | `set_experiments_dir` | 5 |
| `src/debvader/api.py` | create | 5 |
| `src/debvader/data/blends.py` | create from `scripts/real_data/build_blends.py`; variance, mask, splits, injected central | 7 |
| `scripts/real_data/build_blends.py` | thin wrapper | 7 |
| `tests/test_forward_model.py`, `tests/test_measured_noise_loss.py` | create | 5 |

---

## 17. Gotchas

**ceci**

- `resume` does not cascade. After changing an early stage, delete the outputs of every later stage by hand, across all three pipelines.
- A stage that raises leaves `inprogress_*` files behind. They are harmless; delete them when they accumulate.
- All stage modules are imported in both environments. One top-level `import lsst` or `import keras` breaks the other pipeline.
- `Directory.open(path, "w")` deletes the directory. Never use it on a working directory.
- Config keys from `global` are merged flat into every stage's config. Choose global names that cannot collide with stage options.

**Units and normalisation**

- One function builds `x / s`, `s^2 / (f * var)` and `K`. Do not compute any of them elsewhere.
- `stats.json` is computed once, on the isolated training split, and copied. Blend sets and evaluation sets never recompute it.
- The decoder output is in units of `s_b`. Multiply by `s_b` to get nJy. Never apply the inverse `tanh(arcsinh)` to it.
- debvader's `resolve_stats_path` stops if a real build has no `stats.json`. That is deliberate; do not bypass it.

**Geometry**

- Arrays are `[row, column]`; WCS functions return `(x, y) = (column, row)`.
- All stamp and PSF sizes are odd. Centre pixel is `size // 2`.
- `ifftshift` before the FFT, `fftshift` after the inverse FFT. Swapping them is invisible for even sizes and wrong for odd ones.
- Draw with `method="no_pixel"` whenever a profile is convolved with a coadd PSF stamp.
- PSF augmentation: the PSF stamp must receive the same flip and rotation as the image, before `K` is computed.

**Model**

- `KerasMultivariateNormalTriL` samples on every call. Use `mu` from the encoder output for deterministic inference.
- `kl_weight` is driven by the annealing callback toward `kl_target_value`; `loss_weights.kl` is only the initial value.
- The logged `kl_loss` includes the weight.
- Changing `n_out_channels`, the band list or the stamp size changes layer shapes; old weights will not load.
- `BatchNormalization` on the encoder input uses batch statistics in training. Keep the batch size reasonably large (128).

**Data**

- The coadd PSF is constant per 150 px cell. Objects in the same cell share a PSF, which is one more reason to split by patch.
- `INEXACT_PSF` is a mask plane. Consider recording, per stamp, the fraction of pixels carrying it, and check whether kernel-check failures correlate with it.
- Empty-sky windows are rare in a deep field. Count how many each patch yields.
- `$PSCRATCH` is purged. Copy models and metrics to CFS.

**Science caveats to keep in mind when reading results**

- "Isolated" means isolated according to Rubin and Euclid together. A blend neither survey resolves remains in the prior training set.
- The isolation cut removes about 30 % of galaxies and biases the sample; DP2-cutouts keeps the pre-cut catalogue for reweighting.
- PSF model errors map directly into size and shape. A useful robustness test: scale the PSF stamps by a few percent in width at inference and measure how far fluxes, sizes and shapes move.
- Noise correlation is ignored by the L1 likelihood. It makes the data look more constraining than they are, which acts like a slightly too small `beta`.

---

## 18. Open items

- [ ] NERSC account for GPU hours, and the CUDA `jaxlib` install in the `debvader` env.
- [ ] Final sky box: overlap of DP2 ECDFS coverage and the Euclid Q1 footprint.
- [ ] `G_int` FWHM per band, after the full-field PSF histogram. Starting value 0.5 arcsec, Moffat `beta = 4.765`.
- [ ] `kl` weight after the first prior training. Starting value `1/N_pix = 4.79e-5`.
- [ ] `output_activation`: `relu` (start) or `softplus`.
- [ ] Shape labels for blend templates: observed moments (available now) or REGAUSS in the export.
- [ ] Units of `hsm_sigma` expected by `half_light_radius()` in the blend builder.
- [ ] `DEFAULT_B` variance-law slopes: re-measure on the ECDFS coadds.
- [ ] Whether `detect_isPrimary`-type filtering is needed for the Butler `object` table (check for duplicate `objectId` and for parents).
- [ ] Behaviour of `steps_per_epoch = 0` in the existing real-data runs.
- [ ] Name of the Stage 2 best-weights file written by `get_callbacks`.
- [ ] Encoder PSF input (deferred; discuss with Anna before changing the encoder signature).
- [ ] L2 (Fourier) likelihood, using the empty-sky power spectrum or the Butler's `noisecorr` products.
- [ ] Scene-level cutouts and flux redistribution; scarlet baseline on real blends.
- [ ] Joint model: fine model grid (about 0.05 arcsec/px), resampling in `ObservationOperator`, pixel-response handling, Euclid-centred cutouts.
