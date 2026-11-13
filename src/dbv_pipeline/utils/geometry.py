"""Stamp geometry shared by every stage."""


def odd_pixels(size_arcsec: float, pixel_scale: float) -> int:
    """Pixel count for an angular stamp size, always odd so that a centre pixel exists.

    odd_pixels(11.8, 0.2) -> 59;  odd_pixels(11.8, 0.1) -> 119
    """
    n = int(round(size_arcsec / pixel_scale))
    return n if n % 2 == 1 else n + 1


def stamp_geometry(cfg) -> dict:
    """Stamp size in pixels, band count and array shapes from the global configuration."""
    n = odd_pixels(cfg["stamp_arcsec"], cfg["pixel_scale"])
    c = len(cfg["bands"])
    return {"stamp_px": n, "n_bands": c, "input_shape": [n, n, c], "n_pix": n * n * c}
