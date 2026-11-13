"""Translate the pipeline configuration into the configuration objects of the libraries it calls.

The pipeline owns the shared values (bands, pixel scale, stamp size, splits): they are written once, in the
``global`` section, and injected here. A library block that tries to set one of them is an error.
"""

import copy
from pathlib import Path

import yaml

from .geometry import odd_pixels

# Keys of a ``dp2_cutouts:`` block that the pipeline sets itself.
_DP2_INJECTED = ("data_root", "bands", "pixel_scale", "psf_px", "split")
# Keys of ``dp2_cutouts.cuts`` derived from the stamp size.
_DP2_CUTS_INJECTED = ("stamp_px", "margin_px", "diag_px", "aperture_radii")
# Sections DP2-cutouts needs for the catalogue steps (rubin_catalog, euclid_catalog, copaings, catalog_table).
_DP2_REQUIRED = ("name", "sky_box", "rubin", "euclid", "copaings", "cuts")


def build_dp2_cutouts_config(cfg, work_dir, required=_DP2_REQUIRED, label="generated_config"):
    """ceci stage configuration (global section merged in) -> ``dp2_cutouts.config.Config``.

    The DP2-cutouts configuration is assembled from two blocks:
      global.dp2_field    what every DP2-cutouts step needs to know about the field: name, sky_box, butler,
                          rubin, euclid, copaings (written once, shared by SelectSources and MakeCutouts)
      <stage>.dp2_cutouts what only this stage needs (cuts, pixels, masking, export, ...)
    A key may appear in only one of the two.

    ``data_root`` is set to ``work_dir``: the DP2-cutouts tree is a working area, the declared outputs of the
    stage are copied out of it. The generated configuration is also written to
    ``work_dir/generated_config.yaml`` so that a step can be re-run by hand with
    ``dp2-cutouts <that file> <step>`` (one file per `label`, so that two stages do not overwrite each other).
    """
    from dp2_cutouts.config import Config

    if "dp2_cutouts" not in cfg:
        raise KeyError("this stage needs a `dp2_cutouts:` block in its configuration")
    field = copy.deepcopy(dict(cfg.get("dp2_field") or {}))
    stage = copy.deepcopy(dict(cfg["dp2_cutouts"]))
    both = sorted(set(field) & set(stage))
    if both:
        raise ValueError(f"{both} set in both global.dp2_field and the stage's dp2_cutouts block: keep one")
    ceci_cfg = {**field, **stage}

    clash = sorted(set(ceci_cfg) & set(_DP2_INJECTED))
    if clash:
        raise ValueError(f"dp2_cutouts: {clash} are set by the pipeline's global section")
    missing = [k for k in required if k not in ceci_cfg]
    if missing:
        raise KeyError(f"dp2_cutouts: missing {missing}")
    if ceci_cfg["rubin"].get("source", "tap") == "butler" and "butler" not in ceci_cfg:
        raise KeyError("dp2_cutouts.rubin.source is 'butler' but there is no dp2_cutouts.butler block")
    box = ceci_cfg["sky_box"]
    if len(box) != 4 or not (box[0] < box[1] and box[2] < box[3]):
        raise ValueError(f"dp2_cutouts.sky_box must be [ra_min, ra_max, dec_min, dec_max], got {box}")

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    ceci_cfg["data_root"] = str(work_dir)
    ceci_cfg["bands"] = "".join(cfg["bands"])
    ceci_cfg["pixel_scale"] = float(cfg["pixel_scale"])
    ceci_cfg["psf_px"] = int(cfg["psf_stamp_px"])
    ceci_cfg["split"] = dict(cfg["split"])

    cuts = ceci_cfg.setdefault("cuts", {})
    clash = sorted(set(cuts) & set(_DP2_CUTS_INJECTED))
    if clash:
        raise ValueError(f"dp2_cutouts.cuts: {clash} are derived from global.stamp_arcsec")
    n = odd_pixels(cfg["stamp_arcsec"], cfg["pixel_scale"])
    cuts.update(stamp_px=n, margin_px=n // 2, diag_px=2 * (n - 1) + 1, aperture_radii=[n // 2, n - 1])

    path = work_dir / f"{label}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(ceci_cfg, f, sort_keys=False)
    return Config(raw=ceci_cfg, path=path)
