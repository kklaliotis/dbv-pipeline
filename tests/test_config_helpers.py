import pytest

from dbv_pipeline.utils.geometry import odd_pixels, stamp_geometry

GLOBAL = dict(bands=list("ugrizy"), pixel_scale=0.2, stamp_arcsec=11.8, psf_stamp_px=35,
              split=dict(train=0.8, val=0.1, test=0.1, seed=0))
BLOCK = dict(name="t", sky_box=[53.0, 53.3, -27.95, -27.7], rubin=dict(source="tap", tracts=[5063]),
             euclid={}, copaings={}, cuts=dict(snr_min=5.0))


def test_odd_pixels():
    assert odd_pixels(11.8, 0.2) == 59
    assert odd_pixels(11.8, 0.1) == 119
    assert stamp_geometry(GLOBAL)["n_pix"] == 59 * 59 * 6


def test_dp2_config(tmp_path):
    pytest.importorskip("dp2_cutouts")
    from dbv_pipeline.utils.config import build_dp2_cutouts_config

    c = build_dp2_cutouts_config(dict(GLOBAL, dp2_cutouts=BLOCK), tmp_path)
    assert c["cuts"]["stamp_px"] == 59 and c["cuts"]["margin_px"] == 29 and c["cuts"]["diag_px"] == 117
    assert c.sky_box == [53.0, 53.3, -27.95, -27.7] and (tmp_path / "generated_config.yaml").exists()

    bad = {k: v for k, v in BLOCK.items() if k != "sky_box"}
    with pytest.raises(KeyError, match="sky_box"):
        build_dp2_cutouts_config(dict(GLOBAL, dp2_cutouts=dict(bad, skybox=[0, 1, 0, 1])), tmp_path)
    with pytest.raises(ValueError, match="global"):
        build_dp2_cutouts_config(dict(GLOBAL, dp2_cutouts=dict(BLOCK, bands="gri")), tmp_path)


def test_dp2_field_is_shared(tmp_path):
    pytest.importorskip("dp2_cutouts")
    from dbv_pipeline.utils.config import build_dp2_cutouts_config

    field = {k: v for k, v in BLOCK.items() if k != "cuts"}
    c = build_dp2_cutouts_config(dict(GLOBAL, dp2_field=field, dp2_cutouts={"cuts": {"snr_min": 5.0}}), tmp_path,
                                 label="config_test")
    assert c.name == "t" and c["cuts"]["snr_min"] == 5.0 and (tmp_path / "config_test.yaml").exists()
    with pytest.raises(ValueError, match="both"):
        build_dp2_cutouts_config(dict(GLOBAL, dp2_field=field, dp2_cutouts=dict(BLOCK)), tmp_path)
