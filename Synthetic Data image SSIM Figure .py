import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.signal import convolve
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

plt.style.use("fast")
plt.rcParams.update({
"font.size": 6,
"axes.titlesize": 7,
"axes.labelsize": 6,
"xtick.labelsize": 5,
"ytick.labelsize": 5,
"legend.fontsize": 5,
})

# ============================================================
# CONSTANTS
# ============================================================
DT_US = 10.0
INTERNAL_GATE_HYSTERESIS = 0.50
MIN_RASTER_STEP_UM = 0.05

TARGETS_PERCENT = (25.0, 100.0, 200.0)
TARGET_PERCENT_TOL_START = 5.0
TARGET_PERCENT_TOL_STEP = 5.0
TARGET_PERCENT_TOL_MAX = 60.0
TARGET_MIN_PIXELS = 8

CONC_CLASSES = ["Ultra trace", "Trace", "Minor", "Major"]

SHAPE_PRESETS = {
"compact": {"rise_frac": 0.06, "slow_frac": 0.05, "slow_tau_mult": 1.4},
"broad": {"rise_frac": 0.45, "slow_frac": 0.10, "slow_tau_mult": 1.6},
"tail-heavy": {"rise_frac": 0.05, "slow_frac": 0.50, "slow_tau_mult": 6.0},
}
SHAPE_ORDER = ["compact", "broad", "tail-heavy"]

# ============================================================
# CORE SHARED MODEL HELPERS
# ============================================================
def rsd_percent(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    m = float(np.mean(x))
    if m <= 0:
        return np.nan
    s = float(np.std(x, ddof=1))
    return 100.0 * s / m

def finite_data_range(arr):
    arr = np.asarray(arr, dtype=float)
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return 1.0
    return max(float(np.max(vals) - np.min(vals)), 1e-12)

def spot_area_um2_from_diameter(spot_um):
    r = max(float(spot_um), 0.0) / 2.0
    return np.pi * r * r

def shot_volume_um3(spot_um, depth_um):
    return spot_area_um2_from_diameter(spot_um) * max(float(depth_um), 0.0)

def sensitivity_constant_from_anchor(ref_cps, ref_conc_ppm, ref_rep_hz, ref_spot_um, ref_depth_um):
    v_ref = shot_volume_um3(ref_spot_um, ref_depth_um)
    denom = max(float(ref_conc_ppm) * v_ref * float(ref_rep_hz), 1e-30)
    return float(ref_cps) / denom

def class_concentration_map(ultra_trace_ppm, trace_ppm, minor_ppm, major_ppm):
    return {
    "Ultra trace": max(0.0, float(ultra_trace_ppm)),
    "Trace": max(0.0, float(trace_ppm)),
    "Minor": max(0.0, float(minor_ppm)),
    "Major": max(0.0, float(major_ppm)),
    }

def counts_per_um3_at_100_from_anchor_and_class(
    class_name,
    ultra_trace_ppm, trace_ppm, minor_ppm, major_ppm,
    ref_cps, ref_conc_ppm, ref_rep_hz, ref_spot_um, ref_depth_um,
    ):
    cmap = class_concentration_map(ultra_trace_ppm, trace_ppm, minor_ppm, major_ppm)
    conc_ppm = cmap[class_name]
    K = sensitivity_constant_from_anchor(ref_cps, ref_conc_ppm, ref_rep_hz, ref_spot_um, ref_depth_um)
    return K * conc_ppm

def counts_per_shot_from_anchor_and_class(
    class_name,
    ultra_trace_ppm, trace_ppm, minor_ppm, major_ppm,
    ref_cps, ref_conc_ppm, ref_rep_hz, ref_spot_um, ref_depth_um,
    spot_um, depth_um,
    ):
    counts_per_um3 = counts_per_um3_at_100_from_anchor_and_class(
    class_name,
    ultra_trace_ppm, trace_ppm, minor_ppm, major_ppm,
    ref_cps, ref_conc_ppm, ref_rep_hz, ref_spot_um, ref_depth_um,
    )
    return counts_per_um3 * shot_volume_um3(spot_um, depth_um)

def signal_cps_from_anchor_and_class(
    class_name, rep_hz,
    ultra_trace_ppm, trace_ppm, minor_ppm, major_ppm,
    ref_cps, ref_conc_ppm, ref_rep_hz, ref_spot_um, ref_depth_um,
    spot_um, depth_um,
    ):
    cpshot = counts_per_shot_from_anchor_and_class(
    class_name,
    ultra_trace_ppm, trace_ppm, minor_ppm, major_ppm,
    ref_cps, ref_conc_ppm, ref_rep_hz, ref_spot_um, ref_depth_um,
    spot_um, depth_um,
    )
    return cpshot * max(float(rep_hz), 0.0)

def sample_positive_factors(rsd_pct, n, rng):
    rsd = max(0.0, float(rsd_pct)) / 100.0
    if n <= 0:
        return np.array([], dtype=float)
    if rsd <= 0:
        return np.ones(n, dtype=float)
    sigma2 = np.log1p(rsd * rsd)
    mu = -0.5 * sigma2
    return rng.lognormal(mean=mu, sigma=np.sqrt(sigma2), size=n).astype(float)

def shape_params(shape_name):
    shape_name = str(shape_name).strip().lower()
    if shape_name not in SHAPE_PRESETS:
        shape_name = "broad"
    p = SHAPE_PRESETS[shape_name]
    return float(p["rise_frac"]), float(p["slow_frac"]), float(p["slow_tau_mult"])

def washout_kernel_same_model(washout_ms, dt_sec, shape_name):
    washout_ms = max(float(washout_ms), 1e-6)
    rise_frac, slow_frac, slow_tau_mult = shape_params(shape_name)

    tau_fast_s = max((washout_ms / 6.0) / 1000.0, dt_sec)
    tau_slow_s = max(tau_fast_s * slow_tau_mult, dt_sec)
    tau_rise_s = max((washout_ms * rise_frac) / 1000.0, dt_sec)

    kernel_duration_s = max(6.0 * tau_slow_s, 4.0 * washout_ms / 1000.0)
    t_kernel = np.arange(0.0, kernel_duration_s + dt_sec, dt_sec)

    rise_term = 1.0 - np.exp(-t_kernel / tau_rise_s)
    tail = (1.0 - slow_frac) * np.exp(-t_kernel / tau_fast_s) + slow_frac * np.exp(-t_kernel / tau_slow_s)
    kernel = np.maximum(rise_term * tail, 0.0)

    if np.all(kernel <= 0):
        kernel = np.zeros_like(t_kernel)
        kernel[0] = 1.0

    return t_kernel, kernel
def bkg_mu_series(n_bins, Tint_s, bkg_rate_cps, bkg_drift_rsd_percent, bkg_corr_ms, rng):
    base_mu = max(0.0, float(bkg_rate_cps)) * max(float(Tint_s), 0.0)
    out = np.full(int(n_bins), base_mu, dtype=float)

    drift_rsd = max(0.0, float(bkg_drift_rsd_percent)) / 100.0
    corr_ms = max(0.0, float(bkg_corr_ms))

    if n_bins < 3 or base_mu <= 0 or drift_rsd <= 0:
        return out

    corr_s = corr_ms / 1000.0
    sigma_bins = max(1.0, corr_s / max(Tint_s, 1e-12))

    white = rng.normal(0.0, 1.0, size=n_bins)
    hw = max(3, int(np.ceil(4.0 * sigma_bins)))
    x = np.arange(-hw, hw + 1, dtype=float)
    ker = np.exp(-0.5 * (x / sigma_bins) ** 2)
    ker /= np.sum(ker)

    smooth = np.convolve(white, ker, mode="same")
    s = float(np.std(smooth, ddof=1)) if smooth.size > 1 else 0.0
    smooth = smooth / s if s > 0 else np.zeros_like(smooth)

    sigma_ln2 = np.log1p(drift_rsd * drift_rsd)
    sigma_ln = np.sqrt(sigma_ln2)
    mu_ln = -0.5 * sigma_ln2
    mod = np.exp(mu_ln + sigma_ln * smooth)

    return base_mu * mod
def gate_thresholds(mu_bkg, sigma_mult):
    mu_bkg = np.asarray(mu_bkg, dtype=float)
    sig = np.sqrt(np.maximum(mu_bkg, 0.0))
    th_hi = mu_bkg + float(sigma_mult) * sig
    th_lo = mu_bkg + INTERNAL_GATE_HYSTERESIS * float(sigma_mult) * sig
    return th_hi, th_lo

def runs_from_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    x = mask.astype(np.int8)
    d = np.diff(np.r_[0, x, 0])
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts, ends))

def apply_gate_clustered(obs_counts, mu_bkg, gate_mode, sigma_mult, min_run_bins):
    if int(gate_mode) == 0:
        return np.asarray(obs_counts, dtype=float)

    obs = np.asarray(obs_counts, dtype=float)
    mu_bkg = np.asarray(mu_bkg, dtype=float)
    if mu_bkg.size == 1:
        mu_bkg = np.full(obs.size, float(mu_bkg[0]), dtype=float)

    th_hi, th_lo = gate_thresholds(mu_bkg, sigma_mult)
    above_hi = obs >= th_hi
    if not np.any(above_hi):
        return np.zeros_like(obs, dtype=float)

    min_run = max(1, int(min_run_bins))
    hi_runs = runs_from_mask(above_hi)
    detected = np.zeros_like(above_hi, dtype=bool)

    for s, e in hi_runs:
        if (e - s + 1) < min_run:
            continue
        ss, ee = s, e
        while ss - 1 >= 0 and obs[ss - 1] >= th_lo[ss - 1]:
            ss -= 1
        while ee + 1 < obs.size and obs[ee + 1] >= th_lo[ee + 1]:
            ee += 1
        detected[ss:ee + 1] = True

    if not np.any(detected):
        return np.zeros_like(obs, dtype=float)

    if int(gate_mode) == 1:
        return np.where(detected, obs, 0.0)

    out = np.where(detected, obs - mu_bkg, 0.0)
    return np.maximum(out, 0.0)
# ============================================================
# IMAGE / GEOMETRY HELPERS
# ============================================================
def hex_mask(nx=2000, ny=2000, pixel_um=1.0, hex_flat_to_flat_um=2000.0):
    x = (np.arange(nx) - (nx - 1) / 2) * pixel_um
    y = (np.arange(ny) - (ny - 1) / 2) * pixel_um
    X, Y = np.meshgrid(x, y, indexing="xy")

    R = hex_flat_to_flat_um / np.sqrt(3)
    mask = (
        (np.abs(X) <= R)
        & (np.abs(np.sqrt(3) * Y + X) <= 2 * R)
        & (np.abs(np.sqrt(3) * Y - X) <= 2 * R)
    )
    return mask, X, Y
def distance_to_hex_rim(X, Y, hex_flat_to_flat_um=2000.0):
    R = hex_flat_to_flat_um / np.sqrt(3)
    m1 = R - np.abs(X)
    m2 = (2 * R - np.abs(np.sqrt(3) * Y + X)) / 2.0
    m3 = (2 * R - np.abs(np.sqrt(3) * Y - X)) / 2.0
    return np.minimum(m1, np.minimum(m2, m3))

def make_oscillatory_zoned_hex(
    nx=1000,
    ny=1000,
    pixel_um=1.0,
    hex_flat_to_flat_um=1000.0,
    matrix_percent=100.0,
    garnet_default_percent=100.0,
    bands=None,
    center_fill_band_um=5.0,
    center_fill_gap_um=20.0,
    center_fill_lo=10.0,
    center_fill_hi=200.0,
    ):
    mask, X, Y = hex_mask(nx, ny, pixel_um=pixel_um, hex_flat_to_flat_um=hex_flat_to_flat_um)
    d = distance_to_hex_rim(X, Y, hex_flat_to_flat_um=hex_flat_to_flat_um)

    img = np.full((ny, nx), float(matrix_percent), dtype=float)
    img[mask] = float(garnet_default_percent)

    cum = 0.0
    if bands is not None:
        for thickness_um, intensity in bands:
            lo = cum
            hi = cum + float(thickness_um)
            band_mask = mask & (d >= lo) & (d < hi)
            img[band_mask] = float(intensity)
            cum = hi

    dmax = float(np.max(d[mask])) if np.any(mask) else 0.0
    if dmax > cum + center_fill_band_um:
        pitch = float(center_fill_band_um + center_fill_gap_um)
        starts = np.arange(cum, dmax, pitch)
        if len(starts) > 0:
            vals = np.linspace(float(center_fill_lo), float(center_fill_hi), len(starts))
            for lo, inten in zip(starts, vals):
                hi = min(lo + float(center_fill_band_um), dmax + 1e-9)
                band_mask = mask & (d >= lo) & (d < hi)
                img[band_mask] = float(inten)

    return img, mask, X, Y, d
def block_sum(image: np.ndarray, block: int) -> np.ndarray:
    h, w = image.shape
    block = max(int(block), 1)
    h2 = (h // block) * block
    w2 = (w // block) * block
    img = image[:h2, :w2]
    out = img.reshape(h2 // block, block, w2 // block, block).sum(axis=(1, 3))
    return out

def block_mean(image: np.ndarray, block: int) -> np.ndarray:
    return block_sum(image, block) / float(max(int(block), 1) ** 2)

def add_scale_bar(ax, total_width_um, total_height_um=None):
    if total_width_um <= 0:
        return
    if total_height_um is None:
        total_height_um = total_width_um

    target = 0.20 * total_width_um
    bases = np.array([1.0, 2.0, 5.0], dtype=float)[:, None]
    powers = (10.0 ** np.arange(-1, 7, dtype=float))[None, :]
    nice_vals = (bases * powers).ravel()
    bar_um = float(nice_vals[np.argmin(np.abs(np.log10(nice_vals) - np.log10(target)))])

    x0 = 0.06 * total_width_um
    y0 = 0.08 * total_height_um
    lw = 3.0

    ax.plot([x0, x0 + bar_um], [y0, y0], color="white", lw=lw, solid_capstyle="butt")
    ax.plot([x0, x0 + bar_um], [y0, y0], color="black", lw=max(lw - 1.0, 1.0), solid_capstyle="butt")

    label = f"{bar_um/1000.0:g} mm" if bar_um >= 1000 else f"{bar_um:g} µm"
    ax.text(
        x0, y0 + 0.03 * total_height_um, label,
        color="white", fontsize=7, ha="left", va="bottom",
        bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.6),
    )
def safe_percent_reconstruction(obs, resp100):
    obs = np.asarray(obs, dtype=float)
    resp100 = np.asarray(resp100, dtype=float)
    out = np.full_like(obs, np.nan, dtype=float)
    good = np.isfinite(obs) & np.isfinite(resp100) & (resp100 > 0)
    out[good] = 100.0 * obs[good] / resp100[good]
    return out

def safe_ratio(obs, expected):
    obs = np.asarray(obs, dtype=float)
    expected = np.asarray(expected, dtype=float)
    out = np.full_like(obs, np.nan, dtype=float)
    good = np.isfinite(obs) & np.isfinite(expected) & (expected > 0)
    out[good] = obs[good] / expected[good]
    return out

def masked_ssim_against_truth(truth, test):
    truth = np.asarray(truth, dtype=float)
    test = np.asarray(test, dtype=float)

    valid = np.isfinite(truth) & np.isfinite(test)
    if np.sum(valid) < 16:
        return np.nan

    truth_filled = np.where(np.isfinite(truth), truth, 0.0)
    test_filled = np.where(valid, test, truth_filled)

    return ssim(
        truth_filled,
        test_filled,
        data_range=finite_data_range(truth[valid]),
    )
def format_rsd_with_n(rsd_val, n):
    return f"{rsd_val:.2f}% (n={n})" if np.isfinite(rsd_val) else f"-- (n={n})"

def single_pixel_counting_rsd_from_mean_counts(mean_counts):
    if not np.isfinite(mean_counts) or mean_counts <= 0:
        return np.nan
    return 100.0 / np.sqrt(mean_counts)

def target_sample_stats(
    sampled_obs,
    sampled_true_percent,
    targets=TARGETS_PERCENT,
    tol_start=TARGET_PERCENT_TOL_START,
    tol_step=TARGET_PERCENT_TOL_STEP,
    tol_max=TARGET_PERCENT_TOL_MAX,
    min_pixels=TARGET_MIN_PIXELS,
    ):
    sampled_obs = np.asarray(sampled_obs, dtype=float)
    sampled_true_percent = np.asarray(sampled_true_percent, dtype=float)

    finite_mask = np.isfinite(sampled_obs) & np.isfinite(sampled_true_percent)
    out = {}

    for target in targets:
        chosen = None
        tol = tol_start

        while tol <= tol_max + 1e-12:
            m = finite_mask & (sampled_true_percent >= target - tol) & (sampled_true_percent <= target + tol)
            vals = sampled_obs[m]
            true_vals = sampled_true_percent[m]
            if vals.size >= min_pixels:
                chosen = (tol, vals, true_vals)
                break
            tol += tol_step

        if chosen is None:
            all_true = sampled_true_percent[finite_mask]
            all_obs = sampled_obs[finite_mask]
            if all_true.size >= min_pixels:
                order = np.argsort(np.abs(all_true - target))[:min_pixels]
                vals = all_obs[order]
                true_vals = all_true[order]
                tol_used = float(np.max(np.abs(true_vals - target)))
                chosen = (tol_used, vals, true_vals)

        if chosen is None:
            out[target] = {
                "pixel_rsd_percent": np.nan,
                "n_pixels": 0,
                "mean_counts": np.nan,
                "mean_true_percent": np.nan,
                "tol_used": np.nan,
                "single_pixel_counting_rsd_percent": np.nan,
            }
        else:
            tol_used, vals, true_vals = chosen
            mean_counts = float(np.mean(vals)) if vals.size > 0 else np.nan
            out[target] = {
                "pixel_rsd_percent": rsd_percent(vals),
                "n_pixels": int(vals.size),
                "mean_counts": mean_counts,
                "mean_true_percent": float(np.mean(true_vals)) if true_vals.size > 0 else np.nan,
                "tol_used": float(tol_used),
                "single_pixel_counting_rsd_percent": single_pixel_counting_rsd_from_mean_counts(mean_counts),
            }

    return out
# ============================================================
# 1D CENTERLINE TRANSIENT
# ============================================================
def simulate_centerline_binned_counts(
    centerline_percent: np.ndarray,
    counts_per_um3_at_100: float,
    rep_hz: float,
    sample_box_um: float,
    spot_um: float,
    dose: float,
    washout_ms: float,
    Tint_ms: float,
    shot_yield_rsd_percent: float,
    transport_rsd_percent: float,
    shot_jitter_us: float,
    shape_name: str,
    bkg_rate_cps: float,
    bkg_sigma_mult: float,
    bkg_gate_mode: int,
    bkg_min_run_bins: int,
    bkg_drift_rsd_percent: float,
    bkg_corr_ms: float,
    depth_um: float,
    rng: np.random.Generator,
    ):
    rep_hz = max(float(rep_hz), 1e-9)
    washout_ms = max(float(washout_ms), 1e-6)
    Tint_s = max(float(Tint_ms) / 1000.0, 1e-9)
    sample_box_um = max(float(sample_box_um), 1.0)
    spot_um = max(float(spot_um), 1.0)
    dose = max(float(dose), 1.0)

    step_um = max(MIN_RASTER_STEP_UM, sample_box_um / dose)
    Tshot = 1.0 / rep_hz

    centerline_percent = np.asarray(centerline_percent, float)
    n_um = centerline_percent.size

    shot_centers_um = np.arange(0.0, n_um, step_um)
    if shot_centers_um.size == 0:
        return np.array([]), np.array([]), np.array([]), step_um, np.array([]), np.array([]), np.array([])

    half = spot_um / 2.0
    shot_vol_um3 = shot_volume_um3(spot_um, depth_um)

    mu_per_shot = np.zeros(shot_centers_um.size, dtype=float)
    for i, xc in enumerate(shot_centers_um):
        left = max(0, int(np.floor(xc - half)))
        right = min(n_um, int(np.ceil(xc + half)))
        if right <= left:
            idx = int(np.clip(round(xc), 0, n_um - 1))
            local_percent = centerline_percent[idx]
        else:
            local_percent = float(np.mean(centerline_percent[left:right]))

        mu_per_shot[i] = float(counts_per_um3_at_100) * shot_vol_um3 * (local_percent / 100.0)

    shot_factors = (
        sample_positive_factors(shot_yield_rsd_percent, mu_per_shot.size, rng)
        * sample_positive_factors(transport_rsd_percent, mu_per_shot.size, rng)
    )
    mu_per_shot = mu_per_shot * shot_factors

    dt_sec = DT_US / 1e6
    duration = max(mu_per_shot.size * Tshot, Tint_s)
    n_hi = max(2, int(np.ceil(duration / dt_sec)))

    triggers = np.zeros(n_hi, dtype=float)
    shot_times = np.arange(mu_per_shot.size) * Tshot
    if shot_jitter_us > 0:
        shot_times = shot_times + rng.normal(0.0, shot_jitter_us * 1e-6, size=shot_times.size)
    shot_times = np.clip(shot_times, 0.0, max(duration - dt_sec, 0.0))

    shot_idx = np.clip((shot_times / dt_sec).astype(int), 0, n_hi - 1)
    np.add.at(triggers, shot_idx, mu_per_shot)

    _, kernel = washout_kernel_same_model(washout_ms=washout_ms, dt_sec=dt_sec, shape_name=shape_name)
    pulse_area = float(np.sum(kernel) * dt_sec)
    if pulse_area <= 0:
        pulse_area = 1.0

    flux = convolve(triggers, kernel, mode="full")[:n_hi] / pulse_area

    edges = np.arange(0.0, duration + Tint_s, Tint_s)
    if edges.size < 3:
        edges = np.array([0.0, Tint_s, 2.0 * Tint_s])

    cumsum = np.r_[0.0, np.cumsum(flux) * dt_sec]
    sample_edges = np.arange(0.0, n_hi + 1) * dt_sec
    edge_idx = np.searchsorted(sample_edges, edges, side="left")
    edge_idx = np.clip(edge_idx, 0, cumsum.size - 1)

    mu_sig_bins = np.diff(cumsum[edge_idx])
    mu_sig_bins = np.maximum(mu_sig_bins, 0.0)

    mu_bkg_series = bkg_mu_series(
        n_bins=mu_sig_bins.size,
        Tint_s=Tint_s,
        bkg_rate_cps=bkg_rate_cps,
        bkg_drift_rsd_percent=bkg_drift_rsd_percent,
        bkg_corr_ms=bkg_corr_ms,
        rng=rng,
    )

    mu_total_bins = mu_sig_bins + mu_bkg_series
    obs_raw = rng.poisson(mu_total_bins).astype(float)

    obs_processed = apply_gate_clustered(
        obs_counts=obs_raw,
        mu_bkg=mu_bkg_series,
        gate_mode=bkg_gate_mode,
        sigma_mult=bkg_sigma_mult,
        min_run_bins=bkg_min_run_bins,
    )

    t_bins = (edges[:-1] + edges[1:]) / 2.0
    bkg_only_bins = rng.poisson(mu_bkg_series).astype(float)
    threshold_counts = mu_bkg_series + bkg_sigma_mult * np.sqrt(np.maximum(mu_bkg_series, 0.0))

    return (
        t_bins,
        mu_sig_bins,
        obs_processed,
        step_um,
        bkg_only_bins,
        mu_bkg_series,
        threshold_counts,
    )
# ============================================================
# 2D RASTER
# ============================================================
def build_continuous_overlap_image(
    img_percent_1um,
    counts_per_um3_at_100,
    rep_hz,
    dose,
    spot_um,
    sample_box_um,
    shot_yield_rsd_percent,
    transport_rsd_percent,
    bkg_rate_cps,
    bkg_sigma_mult,
    bkg_gate_mode,
    bkg_min_run_bins,
    bkg_drift_rsd_percent,
    bkg_corr_ms,
    depth_um,
    rng,
    ):
    img_percent_1um = np.asarray(img_percent_1um, dtype=float)
    ny, nx = img_percent_1um.shape

    rep_hz = max(float(rep_hz), 1e-9)
    dose = max(float(dose), 1.0)
    spot_um = max(float(spot_um), 1.0)
    sample_box_um = max(float(sample_box_um), 1.0)

    step_um = max(MIN_RASTER_STEP_UM, sample_box_um / dose)
    line_spacing_um = step_um

    obs_counts_1um = np.zeros((ny, nx), dtype=float)
    exp_counts_1um = np.zeros((ny, nx), dtype=float)
    raw_counts_1um = np.zeros((ny, nx), dtype=float)
    bkg_counts_1um = np.zeros((ny, nx), dtype=float)

    half = spot_um / 2.0
    y_centers = np.arange(0.0, ny, line_spacing_um)
    shots_total = 0

    shot_time_s = 1.0 / rep_hz
    shot_bkg_mu_base = bkg_rate_cps * shot_time_s

    shot_bkg_mod = None
    if bkg_drift_rsd_percent > 0:
        approx_total_shots = sum(len(np.arange(0.0, nx, step_um)) for _ in y_centers)
        if approx_total_shots > 0:
            shot_bkg_mod = bkg_mu_series(
                n_bins=approx_total_shots,
                Tint_s=shot_time_s,
                bkg_rate_cps=bkg_rate_cps,
                bkg_drift_rsd_percent=bkg_drift_rsd_percent,
                bkg_corr_ms=bkg_corr_ms,
                rng=rng,
            )
    shot_counter = 0

    for yc in y_centers:
        y0 = max(0, int(np.floor(yc - half)))
        y1 = min(ny, int(np.ceil(yc + half)))
        if y1 <= y0:
            continue

        x_centers = np.arange(0.0, nx, step_um)
        shots_total += x_centers.size

        shot_factors = (
            sample_positive_factors(shot_yield_rsd_percent, x_centers.size, rng)
            * sample_positive_factors(transport_rsd_percent, x_centers.size, rng)
        )

        for i, xc in enumerate(x_centers):
            x0 = max(0, int(np.floor(xc - half)))
            x1 = min(nx, int(np.ceil(xc + half)))
            if x1 <= x0:
                shot_counter += 1
                continue

            patch = img_percent_1um[y0:y1, x0:x1]
            if patch.size == 0:
                shot_counter += 1
                continue

            mean_percent = float(np.mean(patch))
            ncover = patch.size
            shot_vol_um3 = ncover * depth_um

            mu_shot_sig_exp = float(counts_per_um3_at_100) * shot_vol_um3 * (mean_percent / 100.0)
            mu_shot_sig_obs = mu_shot_sig_exp * shot_factors[i]

            if shot_bkg_mod is not None and shot_counter < shot_bkg_mod.size:
                mu_bkg = float(shot_bkg_mod[shot_counter])
            else:
                mu_bkg = float(shot_bkg_mu_base)

            raw_total = float(rng.poisson(max(mu_shot_sig_obs + mu_bkg, 0.0)))

            processed_total = raw_total
            if int(bkg_gate_mode) != 0:
                th_hi = mu_bkg + bkg_sigma_mult * np.sqrt(max(mu_bkg, 0.0))
                if raw_total < th_hi:
                    processed_total = 0.0
                elif int(bkg_gate_mode) == 2:
                    processed_total = max(raw_total - mu_bkg, 0.0)

            exp_share = mu_shot_sig_exp / ncover
            raw_share = raw_total / ncover
            obs_share = processed_total / ncover
            bkg_share = mu_bkg / ncover

            exp_counts_1um[y0:y1, x0:x1] += exp_share
            raw_counts_1um[y0:y1, x0:x1] += raw_share
            obs_counts_1um[y0:y1, x0:x1] += obs_share
            bkg_counts_1um[y0:y1, x0:x1] += bkg_share

            shot_counter += 1

    raster_time_s = shots_total / rep_hz

    return {
        "expected_1um": exp_counts_1um,
        "observed_1um": obs_counts_1um,
        "raw_observed_1um": raw_counts_1um,
        "background_1um": bkg_counts_1um,
        "step_um": step_um,
        "line_spacing_um": line_spacing_um,
        "shots_total": shots_total,
        "raster_time_s": raster_time_s,
        "shot_bkg_mu": shot_bkg_mu_base,
    }
# ============================================================
# UI
# ============================================================
class GarnetImageSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Hex-Garnet Imaging → Sampling → TOF Time-series")
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w = min(int(sw * 0.96), 1900)
        h = min(int(sh * 0.92), 1200)
        self.root.geometry(f"{w}x{h}+10+10")

        self.pixel_um = 1.0
        self.is_running = False
        self._cbar_handles = []

        # Imaging / raster
        self.image_size_mm_var = tk.DoubleVar(value=1.0)
        self.rep_hz_var = tk.DoubleVar(value=1000.0)
        self.wash_ms_var = tk.DoubleVar(value=2.0)
        self.tint_ms_var = tk.DoubleVar(value=0.10)
        self.dose_var = tk.DoubleVar(value=10.0)
        self.spot_um_var = tk.DoubleVar(value=10.0)
        self.sample_box_um_var = tk.IntVar(value=30)
        self.depth_per_shot_um_var = tk.DoubleVar(value=0.1)

        # Sensitivity anchor
        self.ref_conc_ppm_var = tk.DoubleVar(value=500.0)
        self.ref_cps_var = tk.DoubleVar(value=300000.0)
        self.ref_rep_hz_var = tk.DoubleVar(value=200.0)
        self.ref_spot_um_var = tk.DoubleVar(value=5.0)
        self.ref_depth_um_var = tk.DoubleVar(value=0.1)

        # Class concentrations
        self.ultra_trace_ppm_var = tk.DoubleVar(value=0.1)
        self.trace_ppm_var = tk.DoubleVar(value=10.0)
        self.minor_ppm_var = tk.DoubleVar(value=1000.0)
        self.major_ppm_var = tk.DoubleVar(value=100000.0)
        self.active_class_var = tk.StringVar(value="Trace")

        # Variability / shape
        self.shot_yield_rsd_var = tk.DoubleVar(value=5.0)
        self.transport_rsd_var = tk.DoubleVar(value=0.0)
        self.shot_jitter_us_var = tk.DoubleVar(value=0.0)
        self.shape_profile_var = tk.StringVar(value="broad")

        # Background
        self.bkg_rate_cps_var = tk.DoubleVar(value=50.0)
        self.bkg_sigma_mult_var = tk.DoubleVar(value=3.0)
        self.bkg_gate_mode_var = tk.IntVar(value=2)
        self.bkg_min_run_bins_var = tk.IntVar(value=2)
        self.bkg_drift_rsd_var = tk.DoubleVar(value=0.0)
        self.bkg_corr_ms_var = tk.DoubleVar(value=2.0)

        self.cost_per_min_var = tk.DoubleVar(value=5.0)
        self.seed_var = tk.IntVar(value=1)

        self.base_bands = [
            (10, 200), (10, 100), (5, 200), (10, 100),
            (10, 150), (5, 150), (10, 100), (10, 110),
            (10, 100), (5, 110), (10, 100), (10, 90),
            (10, 100), (5, 90), (10, 100), (10, 50),
            (10, 100), (5, 50), (10, 100), (10, 10),
            (10, 100), (5, 10), (10, 100), (15, 200),
            (15, 150), (15, 110), (15, 90), (15, 50),
            (15, 10),
        ]

        self._build_ui()
        self._draw_empty_state()

        # ----------------------------
        # signal helpers
        # ----------------------------
    def concentration_map(self):
        return class_concentration_map(
            self.ultra_trace_ppm_var.get(),
            self.trace_ppm_var.get(),
            self.minor_ppm_var.get(),
            self.major_ppm_var.get(),
        )

    def current_class_conc_ppm(self):
        return self.concentration_map()[self.active_class_var.get()]

    def current_counts_per_um3_at_100(self):
        return counts_per_um3_at_100_from_anchor_and_class(
            class_name=self.active_class_var.get(),
            ultra_trace_ppm=self.ultra_trace_ppm_var.get(),
            trace_ppm=self.trace_ppm_var.get(),
            minor_ppm=self.minor_ppm_var.get(),
            major_ppm=self.major_ppm_var.get(),
            ref_cps=self.ref_cps_var.get(),
            ref_conc_ppm=self.ref_conc_ppm_var.get(),
            ref_rep_hz=self.ref_rep_hz_var.get(),
            ref_spot_um=self.ref_spot_um_var.get(),
            ref_depth_um=self.ref_depth_um_var.get(),
        )

    def current_counts_per_shot(self):
        return counts_per_shot_from_anchor_and_class(
            class_name=self.active_class_var.get(),
            ultra_trace_ppm=self.ultra_trace_ppm_var.get(),
            trace_ppm=self.trace_ppm_var.get(),
            minor_ppm=self.minor_ppm_var.get(),
            major_ppm=self.major_ppm_var.get(),
            ref_cps=self.ref_cps_var.get(),
            ref_conc_ppm=self.ref_conc_ppm_var.get(),
            ref_rep_hz=self.ref_rep_hz_var.get(),
            ref_spot_um=self.ref_spot_um_var.get(),
            ref_depth_um=self.ref_depth_um_var.get(),
            spot_um=self.spot_um_var.get(),
            depth_um=self.depth_per_shot_um_var.get(),
        )

    def current_signal_cps(self):
        return signal_cps_from_anchor_and_class(
            class_name=self.active_class_var.get(),
            rep_hz=self.rep_hz_var.get(),
            ultra_trace_ppm=self.ultra_trace_ppm_var.get(),
            trace_ppm=self.trace_ppm_var.get(),
            minor_ppm=self.minor_ppm_var.get(),
            major_ppm=self.major_ppm_var.get(),
            ref_cps=self.ref_cps_var.get(),
            ref_conc_ppm=self.ref_conc_ppm_var.get(),
            ref_rep_hz=self.ref_rep_hz_var.get(),
            ref_spot_um=self.ref_spot_um_var.get(),
            ref_depth_um=self.ref_depth_um_var.get(),
            spot_um=self.spot_um_var.get(),
            depth_um=self.depth_per_shot_um_var.get(),
        )

        # ----------------------------
        # UI builders
        # ----------------------------
    def _slider(self, parent, label, var, vmin, vmax, row, col):
        f = ttk.Frame(parent)
        f.grid(row=row, column=col, padx=4, pady=2, sticky="ew")
        ttk.Label(f, text=label, font=("Arial", 8)).grid(row=0, column=0, sticky="w")
        s = ttk.Scale(f, from_=vmin, to=vmax, variable=var, orient=tk.HORIZONTAL)
        s.grid(row=0, column=1, sticky="ew", padx=(4, 4))
        e = ttk.Entry(f, textvariable=var, width=8)
        e.grid(row=0, column=2, sticky="e")
        f.columnconfigure(1, weight=1)

    def _entry(self, parent, label, var, row, col):
        f = ttk.Frame(parent)
        f.grid(row=row, column=col, padx=4, pady=2, sticky="ew")
        ttk.Label(f, text=label, font=("Arial", 8)).grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=var, width=8).grid(row=0, column=1, sticky="e", padx=(4, 0))
        f.columnconfigure(0, weight=1)

    def _make_1ax_tab(self, parent, figsize=(13, 7)):
        frame = tk.Frame(parent, bg="white")
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar_frame = tk.Frame(frame, bg="#ccc")
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, toolbar_frame).update()
        return fig, ax, canvas

    def _make_2ax_tab(self, parent, figsize=(15, 7.4)):
        frame = tk.Frame(parent, bg="white")
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar_frame = tk.Frame(frame, bg="#ccc")
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, toolbar_frame).update()
        return fig, axs, canvas

    def _make_2x2_tab(self, parent, figsize=(15, 10)):
        frame = tk.Frame(parent, bg="white")
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar_frame = tk.Frame(frame, bg="#ccc")
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, toolbar_frame).update()
        return fig, axs.ravel(), canvas

    def _build_ui(self):
        top = tk.Frame(self.root, bg="#f0f0f0", pady=3)
        top.pack(side=tk.TOP, fill=tk.X)

        control = ttk.LabelFrame(top, text="Parameters")
        control.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        r, c = 0, 0
        self._slider(control, "Image size (mm)", self.image_size_mm_var, 0.25, 5.0, r, c); c += 1
        self._slider(control, "Rep rate (Hz)", self.rep_hz_var, 50, 2000, r, c); c += 1
        self._slider(control, "Washout (ms)", self.wash_ms_var, 0.2, 10.0, r, c); c += 1
        self._slider(control, "Integration Tint (ms)", self.tint_ms_var, 0.01, 50.0, r, c); c += 1
        self._slider(control, "Dose (shots/pixel)", self.dose_var, 1.0, 100.0, r, c); c += 1
        self._slider(control, "Spot size (µm)", self.spot_um_var, 1.0, 100.0, r, c); c += 1
        self._slider(control, "Sample box (µm)", self.sample_box_um_var, 1, 200, r, c); c += 1
        self._slider(control, "Depth / shot (µm)", self.depth_per_shot_um_var, 0.01, 1.0, r, c); c += 1

        r, c = 1, 0
        self._entry(control, "Ref conc (ppm)", self.ref_conc_ppm_var, r, c); c += 1
        self._entry(control, "Ref signal (cps)", self.ref_cps_var, r, c); c += 1
        self._entry(control, "Ref rep rate (Hz)", self.ref_rep_hz_var, r, c); c += 1
        self._entry(control, "Ref spot (µm)", self.ref_spot_um_var, r, c); c += 1
        self._entry(control, "Ref depth (µm)", self.ref_depth_um_var, r, c); c += 1

        class_frame = ttk.Frame(control)
        class_frame.grid(row=r, column=c, padx=4, pady=2, sticky="ew")
        ttk.Label(class_frame, text="Signal class", font=("Arial", 8)).pack(side=tk.LEFT)
        ttk.Combobox(
            class_frame,
            textvariable=self.active_class_var,
            values=CONC_CLASSES,
            state="readonly",
            width=12
        ).pack(side=tk.LEFT, padx=(6, 0))
        c += 1

        shape_frame = ttk.Frame(control)
        shape_frame.grid(row=r, column=c, padx=4, pady=2, sticky="ew")
        ttk.Label(shape_frame, text="Shape", font=("Arial", 8)).pack(side=tk.LEFT)
        for lbl in SHAPE_ORDER:
            ttk.Radiobutton(shape_frame, text=lbl, variable=self.shape_profile_var, value=lbl).pack(side=tk.LEFT, padx=(3, 0))

        r, c = 2, 0
        self._entry(control, "Ultra trace (ppm)", self.ultra_trace_ppm_var, r, c); c += 1
        self._entry(control, "Trace (ppm)", self.trace_ppm_var, r, c); c += 1
        self._entry(control, "Minor (ppm)", self.minor_ppm_var, r, c); c += 1
        self._entry(control, "Major (ppm)", self.major_ppm_var, r, c); c += 1

        r, c = 3, 0
        self._slider(control, "Shot yield RSD (%)", self.shot_yield_rsd_var, 0.0, 50.0, r, c); c += 1
        self._slider(control, "Transport RSD (%)", self.transport_rsd_var, 0.0, 50.0, r, c); c += 1
        self._slider(control, "Shot jitter (µs)", self.shot_jitter_us_var, 0.0, 200.0, r, c); c += 1

        r, c = 4, 0
        self._slider(control, "Background (cps)", self.bkg_rate_cps_var, 0.0, 5000.0, r, c); c += 1
        self._slider(control, "Gate σ", self.bkg_sigma_mult_var, 0.0, 10.0, r, c); c += 1

        fminrun = ttk.Frame(control)
        fminrun.grid(row=r, column=c, padx=4, pady=2, sticky="ew")
        ttk.Label(fminrun, text="Min run bins", font=("Arial", 8)).grid(row=0, column=0, sticky="w")
        ttk.Entry(fminrun, textvariable=self.bkg_min_run_bins_var, width=6).grid(row=0, column=1, sticky="e", padx=(4, 0))
        fminrun.columnconfigure(0, weight=1)
        c += 1

        self._slider(control, "Bkg drift RSD (%)", self.bkg_drift_rsd_var, 0.0, 100.0, r, c); c += 1
        self._slider(control, "Bkg corr time (ms)", self.bkg_corr_ms_var, 0.1, 20.0, r, c); c += 1

        gate_frame = ttk.Frame(control)
        gate_frame.grid(row=r, column=c, padx=4, pady=2, sticky="ew")
        ttk.Label(gate_frame, text="Gate", font=("Arial", 8)).pack(side=tk.LEFT)
        ttk.Radiobutton(gate_frame, text="Off", variable=self.bkg_gate_mode_var, value=0).pack(side=tk.LEFT, padx=(3, 0))
        ttk.Radiobutton(gate_frame, text="Gate", variable=self.bkg_gate_mode_var, value=1).pack(side=tk.LEFT, padx=(3, 0))
        ttk.Radiobutton(gate_frame, text="Gate+Sub", variable=self.bkg_gate_mode_var, value=2).pack(side=tk.LEFT, padx=(3, 0))
        c += 1

        r, c = 5, 0
        self._slider(control, "$ / minute", self.cost_per_min_var, 0.0, 100.0, r, c); c += 1

        fseed = ttk.Frame(control)
        fseed.grid(row=r, column=c, padx=4, pady=2, sticky="ew")
        ttk.Label(fseed, text="Seed", font=("Arial", 8)).grid(row=0, column=0, sticky="w")
        ttk.Entry(fseed, textvariable=self.seed_var, width=6).grid(row=0, column=1, sticky="e", padx=(4, 0))
        fseed.columnconfigure(0, weight=1)

        for i in range(10):
            control.columnconfigure(i, weight=1)

        btn = tk.Frame(top, bg="#f0f0f0", pady=2)
        btn.pack(side=tk.TOP, fill=tk.X)

        self.update_button = tk.Button(
            btn, text="UPDATE", command=self.start_update,
            bg="#e1e1e1", font=("Arial", 10, "bold"), height=1, width=16
        )
        self.update_button.pack(side=tk.LEFT, padx=(10, 8))

        self.progress = ttk.Progressbar(btn, orient="horizontal", mode="indeterminate", length=250)
        self.progress.pack(side=tk.LEFT, padx=(8, 8))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(btn, textvariable=self.status_var).pack(side=tk.LEFT, padx=(4, 10))

        stats_outer = tk.Frame(self.root, bg="#e1e4e8")
        stats_outer.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        stats_row1 = tk.Frame(stats_outer, bg="#e1e4e8")
        stats_row1.pack(side=tk.TOP, fill=tk.X)
        stats_row2 = tk.Frame(stats_outer, bg="#e1e4e8")
        stats_row2.pack(side=tk.TOP, fill=tk.X)
        stats_row3 = tk.Frame(stats_outer, bg="#e1e4e8")
        stats_row3.pack(side=tk.TOP, fill=tk.X)

        self.stat_step_var = tk.StringVar(value="--")
        self.stat_time_var = tk.StringVar(value="--")
        self.stat_cost_var = tk.StringVar(value="--")
        self.stat_mse_var = tk.StringVar(value="--")
        self.stat_ssim_var = tk.StringVar(value="--")
        self.stat_class_var = tk.StringVar(value="--")

        self.stat_rsd25_var = tk.StringVar(value="--")
        self.stat_rsd100_var = tk.StringVar(value="--")
        self.stat_rsd200_var = tk.StringVar(value="--")

        self.stat_count25_var = tk.StringVar(value="--")
        self.stat_count100_var = tk.StringVar(value="--")
        self.stat_count200_var = tk.StringVar(value="--")

        self.stat_sp25_var = tk.StringVar(value="--")
        self.stat_sp100_var = tk.StringVar(value="--")
        self.stat_sp200_var = tk.StringVar(value="--")

        def make_stat(parent, label, var, color="#007acc"):
            f = tk.Frame(parent, bg="#e1e4e8", bd=1, relief=tk.RIDGE)
            f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
            tk.Label(f, text=label, bg="#e1e4e8", fg="#555", font=("Arial", 7, "bold")).pack(pady=1)
            tk.Label(
                f, textvariable=var, bg="#e1e4e8", fg=color,
                font=("Arial", 8, "bold"), wraplength=160, justify="center"
            ).pack(pady=1)

        make_stat(stats_row1, "Step / Overlap", self.stat_step_var, color="#009E73")
        make_stat(stats_row1, "Raster time", self.stat_time_var, color="#D55E00")
        make_stat(stats_row1, "Estimated cost", self.stat_cost_var, color="#D55E00")
        make_stat(stats_row1, "MSE (recon→truth)", self.stat_mse_var, color="#6A3D9A")
        make_stat(stats_row1, "SSIM (recon→truth)", self.stat_ssim_var, color="#6A3D9A")
        make_stat(stats_row1, "Active class", self.stat_class_var, color="#444444")

        make_stat(stats_row2, "Pixel RSD near 25%", self.stat_rsd25_var, color="#1B9E77")
        make_stat(stats_row2, "Pixel RSD near 100%", self.stat_rsd100_var, color="#1B9E77")
        make_stat(stats_row2, "Pixel RSD near 200%", self.stat_rsd200_var, color="#1B9E77")
        make_stat(stats_row2, "Mean counts near 25%", self.stat_count25_var, color="#0072B2")
        make_stat(stats_row2, "Mean counts near 100%", self.stat_count100_var, color="#0072B2")
        make_stat(stats_row2, "Mean counts near 200%", self.stat_count200_var, color="#0072B2")

        make_stat(stats_row3, "1/√N near 25%", self.stat_sp25_var, color="#AA4499")
        make_stat(stats_row3, "1/√N near 100%", self.stat_sp100_var, color="#AA4499")
        make_stat(stats_row3, "1/√N near 200%", self.stat_sp200_var, color="#AA4499")

        self.container = tk.Frame(self.root, bg="white", bd=2, relief=tk.SUNKEN)
        self.container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.nb = ttk.Notebook(self.container)
        self.nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.tab_percent = ttk.Frame(self.nb)
        self.nb.add(self.tab_percent, text="Percent Images")
        self.fig_percent, (self.axp1, self.axp2), self.can_percent = self._make_2ax_tab(self.tab_percent)

        self.tab_counts = ttk.Frame(self.nb)
        self.nb.add(self.tab_counts, text="Counts Images")
        self.fig_counts, (self.axc1, self.axc2), self.can_counts = self._make_2ax_tab(self.tab_counts)

        self.tab_center = ttk.Frame(self.nb)
        self.nb.add(self.tab_center, text="Centerline time-series")
        self.fig_center, self.ax_center, self.can_center = self._make_1ax_tab(self.tab_center)

        self.tab_rsdmap = ttk.Frame(self.nb)
        self.nb.add(self.tab_rsdmap, text="Pixel RSD Maps")
        self.fig_rsdmap, (self.axr1, self.axr2), self.can_rsdmap = self._make_2ax_tab(self.tab_rsdmap)

        self.tab_metrics = ttk.Frame(self.nb)
        self.nb.add(self.tab_metrics, text="Metrics")
        self.fig_metrics, (self.axm1, self.axm2, self.axm3, self.axm4), self.can_metrics = self._make_2x2_tab(self.tab_metrics)

    def _draw_empty_state(self):
        for cb in self._cbar_handles:
            try:
                cb.remove()
            except Exception:
                pass
        self._cbar_handles = []

        for ax in [self.axp1, self.axp2, self.axc1, self.axc2, self.ax_center, self.axr1, self.axr2, self.axm1, self.axm2, self.axm3, self.axm4]:
            ax.clear()
            ax.text(0.5, 0.5, "Press UPDATE", transform=ax.transAxes, ha="center", va="center")

        self.can_percent.draw()
        self.can_counts.draw()
        self.can_center.draw()
        self.can_rsdmap.draw()
        self.can_metrics.draw()

    def start_update(self):
        if self.is_running:
            return
        self.is_running = True
        self.update_button.config(state=tk.DISABLED, text="RUNNING...")
        self.status_var.set("Updating plots...")
        self.progress.start(10)
        self.root.after(50, self.update_all)

    def update_all(self):
        try:
            for cb in self._cbar_handles:
                try:
                    cb.remove()
                except Exception:
                    pass
            self._cbar_handles = []

            image_size_mm = max(float(self.image_size_mm_var.get()), 0.25)
            rep_hz = float(self.rep_hz_var.get())
            wash_ms = float(self.wash_ms_var.get())
            Tint_ms = float(self.tint_ms_var.get())
            dose = max(float(self.dose_var.get()), 1.0)
            spot_um = max(float(self.spot_um_var.get()), 1.0)
            box_um = max(float(self.sample_box_um_var.get()), 1.0)
            depth_um = max(float(self.depth_per_shot_um_var.get()), 1e-6)

            counts_per_um3 = max(float(self.current_counts_per_um3_at_100()), 0.0)
            class_name = self.active_class_var.get()
            class_ppm = self.current_class_conc_ppm()
            signal_cps = self.current_signal_cps()
            counts_per_shot = self.current_counts_per_shot()

            shot_yield_rsd = float(self.shot_yield_rsd_var.get())
            transport_rsd = float(self.transport_rsd_var.get())
            shot_jitter_us = float(self.shot_jitter_us_var.get())
            shape_name = str(self.shape_profile_var.get())

            bkg_rate_cps = max(float(self.bkg_rate_cps_var.get()), 0.0)
            sigma_mult = max(float(self.bkg_sigma_mult_var.get()), 0.0)
            gate_mode = int(self.bkg_gate_mode_var.get())
            min_run_bins = int(self.bkg_min_run_bins_var.get())
            bkg_drift_rsd = float(self.bkg_drift_rsd_var.get())
            bkg_corr_ms = float(self.bkg_corr_ms_var.get())

            cost_per_min = max(float(self.cost_per_min_var.get()), 0.0)
            seed = int(self.seed_var.get())

            rng = np.random.default_rng(seed)

            self.nx = max(120, int(round(image_size_mm * 1000.0)))
            self.ny = self.nx
            self.hex_flat_to_flat_um = image_size_mm * 1000.0

            scale_factor = image_size_mm / 1.0
            scaled_bands = [(thk * scale_factor, inten) for thk, inten in self.base_bands]

            box = max(1, int(round(box_um / self.pixel_um)))
            eff_um = box * self.pixel_um

            img_percent, mask, X, Y, d = make_oscillatory_zoned_hex(
                nx=self.nx,
                ny=self.ny,
                pixel_um=self.pixel_um,
                hex_flat_to_flat_um=self.hex_flat_to_flat_um,
                matrix_percent=100.0,
                garnet_default_percent=100.0,
                bands=scaled_bands,
                center_fill_band_um=2.0,
                center_fill_gap_um=20.0 * scale_factor,
                center_fill_lo=10.0,
                center_fill_hi=200.0,
            )

            self.status_var.set("Building observed raster...")
            self.root.update_idletasks()

            raster = build_continuous_overlap_image(
                img_percent_1um=img_percent,
                counts_per_um3_at_100=counts_per_um3,
                rep_hz=rep_hz,
                dose=dose,
                spot_um=spot_um,
                sample_box_um=box_um,
                shot_yield_rsd_percent=shot_yield_rsd,
                transport_rsd_percent=transport_rsd,
                bkg_rate_cps=bkg_rate_cps,
                bkg_sigma_mult=sigma_mult,
                bkg_gate_mode=gate_mode,
                bkg_min_run_bins=min_run_bins,
                bkg_drift_rsd_percent=bkg_drift_rsd,
                bkg_corr_ms=bkg_corr_ms,
                depth_um=depth_um,
                rng=rng,
            )

            self.status_var.set("Building uniform 100% response raster...")
            self.root.update_idletasks()

            uniform_100 = np.full_like(img_percent, 100.0, dtype=float)
            raster_100 = build_continuous_overlap_image(
                img_percent_1um=uniform_100,
                counts_per_um3_at_100=counts_per_um3,
                rep_hz=rep_hz,
                dose=dose,
                spot_um=spot_um,
                sample_box_um=box_um,
                shot_yield_rsd_percent=0.0,
                transport_rsd_percent=0.0,
                bkg_rate_cps=0.0,
                bkg_sigma_mult=sigma_mult,
                bkg_gate_mode=0,
                bkg_min_run_bins=min_run_bins,
                bkg_drift_rsd_percent=0.0,
                bkg_corr_ms=bkg_corr_ms,
                depth_um=depth_um,
                rng=np.random.default_rng(seed + 999),
            )

            mu_1um = raster["expected_1um"]
            obs_1um = raster["observed_1um"]
            raw_1um = raster["raw_observed_1um"]
            bkg_1um = raster["background_1um"]

            response_100_1um = raster_100["expected_1um"]

            sampled_obs = block_sum(obs_1um, block=box)
            sampled_mu = block_sum(mu_1um, block=box)
            sampled_raw = block_sum(raw_1um, block=box)
            sampled_bkg = block_sum(bkg_1um, block=box)

            sampled_response_100 = block_sum(response_100_1um, block=box)
            sampled_true_percent = block_mean(img_percent, block=box)

            recon_percent_1um = safe_percent_reconstruction(obs_1um, response_100_1um)
            sampled_obs_percent = safe_percent_reconstruction(sampled_obs, sampled_response_100)

            true_img_1um = np.asarray(img_percent, dtype=float)
            true_img_sampled_percent = np.asarray(sampled_true_percent, dtype=float)

            full_extent = [0, self.nx * self.pixel_um, 0, self.ny * self.pixel_um]
            sampled_extent = [0, sampled_obs.shape[1] * eff_um, 0, sampled_obs.shape[0] * eff_um]

            step_um = raster["step_um"]
            overlap_um = max(spot_um - step_um, 0.0)
            overlap_pct = 100.0 * overlap_um / spot_um if spot_um > 0 else 0.0
            raster_time_s = raster["raster_time_s"]
            est_cost = (raster_time_s / 60.0) * cost_per_min

            valid_1um = np.isfinite(true_img_1um) & np.isfinite(recon_percent_1um)
            if np.any(valid_1um):
                mse_val = mean_squared_error(true_img_1um[valid_1um], recon_percent_1um[valid_1um])
                ssim_val = masked_ssim_against_truth(true_img_1um, recon_percent_1um)
            else:
                mse_val = np.nan
                ssim_val = np.nan

            self.status_var.set("Estimating target precision statistics...")
            self.root.update_idletasks()

            target_stats = target_sample_stats(
                sampled_obs=sampled_obs,
                sampled_true_percent=sampled_true_percent,
            )

            s25 = target_stats[25.0]
            s100 = target_stats[100.0]
            s200 = target_stats[200.0]

            self.stat_step_var.set(f"{step_um:.3f} µm / {overlap_pct:.1f}%")
            self.stat_time_var.set(f"{raster_time_s:.1f} s")
            self.stat_cost_var.set(f"${est_cost:.2f}")
            self.stat_mse_var.set(f"{mse_val:.5f}" if np.isfinite(mse_val) else "--")
            self.stat_ssim_var.set(f"{ssim_val:.5f}" if np.isfinite(ssim_val) else "--")
            self.stat_class_var.set(
                f"{class_name}\n{class_ppm:g} ppm\n{signal_cps:,.0f} cps\n{counts_per_shot:.2f} cts/shot"
            )

            self.stat_rsd25_var.set(format_rsd_with_n(s25["pixel_rsd_percent"], s25["n_pixels"]))
            self.stat_rsd100_var.set(format_rsd_with_n(s100["pixel_rsd_percent"], s100["n_pixels"]))
            self.stat_rsd200_var.set(format_rsd_with_n(s200["pixel_rsd_percent"], s200["n_pixels"]))

            self.stat_count25_var.set(f"{s25['mean_counts']:.2f}" if np.isfinite(s25["mean_counts"]) else "--")
            self.stat_count100_var.set(f"{s100['mean_counts']:.2f}" if np.isfinite(s100["mean_counts"]) else "--")
            self.stat_count200_var.set(f"{s200['mean_counts']:.2f}" if np.isfinite(s200["mean_counts"]) else "--")

            self.stat_sp25_var.set(f"{s25['single_pixel_counting_rsd_percent']:.2f}%" if np.isfinite(s25["single_pixel_counting_rsd_percent"]) else "--")
            self.stat_sp100_var.set(f"{s100['single_pixel_counting_rsd_percent']:.2f}%" if np.isfinite(s100["single_pixel_counting_rsd_percent"]) else "--")
            self.stat_sp200_var.set(f"{s200['single_pixel_counting_rsd_percent']:.2f}%" if np.isfinite(s200["single_pixel_counting_rsd_percent"]) else "--")

            gate_txt = {0: "Off", 1: "Gate", 2: "Gate+Sub"}.get(gate_mode, "?")

            # ------------------------------------------------
            # TAB 1: Percent Images
            # ------------------------------------------------
            self.axp1.clear()
            self.axp2.clear()

            true_vmin = float(np.nanmin(img_percent))
            true_vmax = float(np.nanmax(img_percent))

            im1 = self.axp1.imshow(
                img_percent, origin="lower", interpolation="nearest",
                extent=full_extent, vmin=true_vmin, vmax=true_vmax
            )
            self.axp1.set_title(
                f"True 1 µm image (% scale)\n"
                f"{class_name}, {class_ppm:g} ppm | density={counts_per_um3:.4g} counts/µm³ @100%"
            )
            self.axp1.set_xlabel("µm")
            self.axp1.set_ylabel("µm")
            add_scale_bar(self.axp1, self.nx * self.pixel_um, self.ny * self.pixel_um)
            self._cbar_handles.append(self.fig_percent.colorbar(im1, ax=self.axp1, fraction=0.046, pad=0.04))

            im2 = self.axp2.imshow(
                sampled_obs_percent, origin="lower", interpolation="nearest",
                extent=sampled_extent, vmin=true_vmin, vmax=true_vmax
            )
            self.axp2.set_title(
                f"Response-map normalized sampled image ({eff_um:.0f} µm boxes)\n"
                f"MSE={mse_val:.5f}, SSIM={ssim_val:.5f} | shape={shape_name}, gate={gate_txt}"
            )
            self.axp2.set_xlabel("µm")
            self.axp2.set_ylabel("µm")
            add_scale_bar(self.axp2, sampled_obs.shape[1] * eff_um, sampled_obs.shape[0] * eff_um)
            self._cbar_handles.append(self.fig_percent.colorbar(im2, ax=self.axp2, fraction=0.046, pad=0.04))
            self.can_percent.draw()

            # ------------------------------------------------
            # TAB 2: Counts Images
            # ------------------------------------------------
            self.axc1.clear()
            self.axc2.clear()

            im3 = self.axc1.imshow(
                sampled_obs, origin="lower", interpolation="nearest", extent=sampled_extent
            )
            self.axc1.set_title(
                f"Observed sampled counts per pixel ({eff_um:.0f} µm boxes)\n"
                f"Raw signal strength after sampling"
            )
            self.axc1.set_xlabel("µm")
            self.axc1.set_ylabel("µm")
            add_scale_bar(self.axc1, sampled_obs.shape[1] * eff_um, sampled_obs.shape[0] * eff_um)
            self._cbar_handles.append(self.fig_counts.colorbar(im3, ax=self.axc1, fraction=0.046, pad=0.04))

            im4 = self.axc2.imshow(
                sampled_mu, origin="lower", interpolation="nearest", extent=sampled_extent
            )
            self.axc2.set_title(
                f"Expected sampled counts per pixel ({eff_um:.0f} µm boxes)\n"
                f"Noise-free reference counts"
            )
            self.axc2.set_xlabel("µm")
            self.axc2.set_ylabel("µm")
            add_scale_bar(self.axc2, sampled_mu.shape[1] * eff_um, sampled_mu.shape[0] * eff_um)
            self._cbar_handles.append(self.fig_counts.colorbar(im4, ax=self.axc2, fraction=0.046, pad=0.04))
            self.can_counts.draw()

            # ------------------------------------------------
            # TAB 3: Centerline time-series
            # ------------------------------------------------
            centerline_percent = img_percent[self.ny // 2, :]

            (
                t_bins,
                mu_sig_bins,
                obs_bins,
                step_um_line,
                bkg_only_bins,
                mu_bkg_counts,
                threshold_counts,
            ) = simulate_centerline_binned_counts(
                centerline_percent=centerline_percent,
                counts_per_um3_at_100=counts_per_um3,
                rep_hz=rep_hz,
                sample_box_um=box_um,
                spot_um=spot_um,
                dose=dose,
                washout_ms=wash_ms,
                Tint_ms=Tint_ms,
                shot_yield_rsd_percent=shot_yield_rsd,
                transport_rsd_percent=transport_rsd,
                shot_jitter_us=shot_jitter_us,
                shape_name=shape_name,
                bkg_rate_cps=bkg_rate_cps,
                bkg_sigma_mult=sigma_mult,
                bkg_gate_mode=gate_mode,
                bkg_min_run_bins=min_run_bins,
                bkg_drift_rsd_percent=bkg_drift_rsd,
                bkg_corr_ms=bkg_corr_ms,
                depth_um=depth_um,
                rng=np.random.default_rng(seed + 10),
            )

            self.ax_center.clear()
            if t_bins.size > 0:
                self.ax_center.plot(t_bins, obs_bins, drawstyle="steps-mid", linewidth=1.0, label="Observed (processed)")
                self.ax_center.plot(t_bins, mu_sig_bins, linewidth=1.0, alpha=0.95, label="Expected signal")
                self.ax_center.plot(t_bins, bkg_only_bins, linewidth=0.8, alpha=0.35, label="Noisy background")
                self.ax_center.plot(t_bins, mu_bkg_counts, color="black", linestyle=":", linewidth=0.9, alpha=0.9, label="Mean background")
                self.ax_center.plot(t_bins, threshold_counts, color="black", linestyle="--", linewidth=0.9, alpha=0.9, label=f"{sigma_mult:g}σ threshold")

                self.ax_center.set_title(
                    f"Centerline bins: rep={rep_hz:g} Hz, wash={wash_ms:g} ms, Tint={Tint_ms:g} ms, "
                    f"pixel={box_um:g} µm, dose={dose:g}, spot={spot_um:g} µm, depth={depth_um:g} µm, step={step_um_line:.3f} µm\n"
                    f"{class_name}, {class_ppm:g} ppm | signal={signal_cps:,.0f} cps | density={counts_per_um3:.4g} counts/µm³ @100%"
                )
                self.ax_center.set_xlabel("Time (s)")
                self.ax_center.set_ylabel("Counts per integration bin")
                self.ax_center.grid(True, alpha=0.25)
                self.ax_center.legend(loc="upper right", frameon=True, fontsize=6)
            else:
                self.ax_center.text(0.5, 0.5, "Not enough bins.", transform=self.ax_center.transAxes, ha="center", va="center")
            self.can_center.draw()

            # ------------------------------------------------
            # TAB 4: Pixel RSD Maps
            # ------------------------------------------------
            self.axr1.clear()
            self.axr2.clear()

            rsd_1um_map = np.full_like(mu_1um, np.nan, dtype=float)
            good1 = mu_1um > 0
            rsd_1um_map[good1] = 100.0 / np.sqrt(mu_1um[good1])

            rsd_sample_map = np.full_like(sampled_mu, np.nan, dtype=float)
            goods = sampled_mu > 0
            rsd_sample_map[goods] = 100.0 / np.sqrt(sampled_mu[goods])

            im5 = self.axr1.imshow(rsd_1um_map, origin="lower", interpolation="nearest", extent=full_extent)
            self.axr1.set_title("Single-pixel counting RSD map at 1 µm (%)")
            self.axr1.set_xlabel("µm")
            self.axr1.set_ylabel("µm")
            add_scale_bar(self.axr1, self.nx * self.pixel_um, self.ny * self.pixel_um)
            self._cbar_handles.append(self.fig_rsdmap.colorbar(im5, ax=self.axr1, fraction=0.046, pad=0.04))

            im6 = self.axr2.imshow(rsd_sample_map, origin="lower", interpolation="nearest", extent=sampled_extent)
            self.axr2.set_title(
                f"Single-pixel counting RSD map at sampled scale ({eff_um:.0f} µm) (%)\n"
                f"1/√N based on expected sampled counts"
            )
            self.axr2.set_xlabel("µm")
            self.axr2.set_ylabel("µm")
            add_scale_bar(self.axr2, sampled_obs.shape[1] * eff_um, sampled_obs.shape[0] * eff_um)
            self._cbar_handles.append(self.fig_rsdmap.colorbar(im6, ax=self.axr2, fraction=0.046, pad=0.04))
            self.can_rsdmap.draw()

            # ------------------------------------------------
            # TAB 5: Metrics
            # ------------------------------------------------
            self.axm1.clear()
            self.axm2.clear()
            self.axm3.clear()
            self.axm4.clear()

            ratio_1um = safe_ratio(obs_1um, mu_1um)
            ratio_sample = safe_ratio(sampled_obs, sampled_mu)

            im7 = self.axm1.imshow(ratio_1um, origin="lower", interpolation="nearest", extent=full_extent, vmin=0.0, vmax=2.0)
            self.axm1.set_title("Observed / expected counts at 1 µm")
            self.axm1.set_xlabel("µm")
            self.axm1.set_ylabel("µm")
            add_scale_bar(self.axm1, self.nx * self.pixel_um, self.ny * self.pixel_um)
            self._cbar_handles.append(self.fig_metrics.colorbar(im7, ax=self.axm1, fraction=0.046, pad=0.04))

            im8 = self.axm2.imshow(ratio_sample, origin="lower", interpolation="nearest", extent=sampled_extent, vmin=0.0, vmax=2.0)
            self.axm2.set_title(f"Observed / expected counts at sampled scale ({eff_um:.0f} µm)")
            self.axm2.set_xlabel("µm")
            self.axm2.set_ylabel("µm")
            add_scale_bar(self.axm2, sampled_obs.shape[1] * eff_um, sampled_obs.shape[0] * eff_um)
            self._cbar_handles.append(self.fig_metrics.colorbar(im8, ax=self.axm2, fraction=0.046, pad=0.04))

            im9 = self.axm3.imshow(sampled_bkg, origin="lower", interpolation="nearest", extent=sampled_extent)
            self.axm3.set_title(f"Mean background counts per sampled pixel ({eff_um:.0f} µm)")
            self.axm3.set_xlabel("µm")
            self.axm3.set_ylabel("µm")
            add_scale_bar(self.axm3, sampled_obs.shape[1] * eff_um, sampled_obs.shape[0] * eff_um)
            self._cbar_handles.append(self.fig_metrics.colorbar(im9, ax=self.axm3, fraction=0.046, pad=0.04))

            im10 = self.axm4.imshow(sampled_true_percent, origin="lower", interpolation="nearest", extent=sampled_extent)
            self.axm4.set_title(
                f"True percent at sampled scale ({eff_um:.0f} µm)\n"
                f"RSD25={format_rsd_with_n(s25['pixel_rsd_percent'], s25['n_pixels'])} | "
                f"RSD100={format_rsd_with_n(s100['pixel_rsd_percent'], s100['n_pixels'])} | "
                f"RSD200={format_rsd_with_n(s200['pixel_rsd_percent'], s200['n_pixels'])}"
            )
            self.axm4.set_xlabel("µm")
            self.axm4.set_ylabel("µm")
            add_scale_bar(self.axm4, sampled_obs.shape[1] * eff_um, sampled_obs.shape[0] * eff_um)
            self._cbar_handles.append(self.fig_metrics.colorbar(im10, ax=self.axm4, fraction=0.046, pad=0.04))

            self.can_metrics.draw()

            self.status_var.set("Done")

        except Exception as e:
            print("ERROR (update_all):", e)
            self.status_var.set(f"Error: {e}")

        finally:
            self.progress.stop()
            self.is_running = False
            self.update_button.config(state=tk.NORMAL, text="UPDATE")
print("Imaging script started")

if __name__ == "__main__":
    root = tk.Tk()
    app = GarnetImageSimulator(root)
    root.mainloop()

