import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.signal import convolve

plt.style.use("fast")

# Keep text and line work editable when figures are exported to PDF.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"


# ============================================================
# CONSTANTS
# ============================================================

DT_US = 10.0

CONC_CLASSES = ["Ultra trace", "Trace", "Minor", "Major"]

SHAPE_PRESETS = {
    "compact": {"rise_frac": 0.06, "slow_frac": 0.05, "slow_tau_mult": 1.4},
    "broad": {"rise_frac": 0.45, "slow_frac": 0.10, "slow_tau_mult": 1.6},
    "tail-heavy": {"rise_frac": 0.05, "slow_frac": 0.50, "slow_tau_mult": 6.0},
}
SHAPE_ORDER = ["compact", "broad", "tail-heavy"]

TOF_TINTS_MS = [0.1, 1.0, 5.0]
QMS_COMPARE_N = [1, 2, 5, 10, 20]

QMS_OVERLAY_COLORS = [
    "C0", "C1", "C2", "C3", "C4",
    "C5", "C6", "C7", "C8", "C9",
]

# Tab 3 only: the three ToF mass windows requested for comparison.
# The count gain is an explicit first-order model: sensitivity is assumed to
# scale inversely with the acquired mass-window span relative to 23–238.
TOF_MASS_WINDOWS = [
    {"label": "ToF 23–238", "low": 23.0, "high": 238.0, "color": "C0", "linestyle": "--"},
    {"label": "ToF 138–238", "low": 138.0, "high": 238.0, "color": "C1", "linestyle": "-."},
    {"label": "ToF 200–238", "low": 200.0, "high": 238.0, "color": "C2", "linestyle": ":"},
]

# Tab 3 defaults. Three explicit QMS sweep conditions are plotted. For each
# sweep, the shaded field spans the entered relative QMS sensitivity bounds,
# while the line uses the current Quad sensitivity multiplier.
TAB3_DEFAULT_QMS_SWEEPS_MS = "5, 10, 20"
TAB3_DEFAULT_QMS_SENSITIVITIES = "1, 20"

# Restricted-range Vitesse work has demonstrated gains up to about fivefold.
# The simple inverse-span estimate is therefore capped at 5x rather than
# allowing the 200-238 window to exceed the demonstrated first-order gain.
TAB3_TOF_MAX_GAIN = 5.0

TAB3_DEFAULT_MIN_DWELL_MS = 0.1

# Visual definitions for the three QMS sweep families. The fields are kept
# transparent so all three can be viewed together on the same x-y graph.
TAB3_QMS_STYLE_CYCLE = [
    {"color": "C4", "linestyle": "-",  "alpha": 0.15, "hatch": None},
    {"color": "C6", "linestyle": "--", "alpha": 0.13, "hatch": None},
    {"color": "0.35", "linestyle": ":", "alpha": 0.11, "hatch": "//"},
]


# ============================================================
# BASIC HELPERS
# ============================================================

def finite_positive(x):
    return np.isfinite(x) and x > 0


def parse_positive_number_list(text, field_name):
    """Parse comma/space/semicolon separated positive numbers."""
    parts = [p for p in re.split(r"[,;\s]+", str(text).strip()) if p]

    if not parts:
        raise ValueError(f"{field_name} must contain at least one positive number.")

    values = []
    for part in parts:
        try:
            value = float(part)
        except ValueError as exc:
            raise ValueError(f"Invalid value '{part}' in {field_name}.") from exc

        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"All values in {field_name} must be positive.")
        values.append(value)

    return sorted(set(values))


def tof_mass_window_multiplier(
    low_mass,
    high_mass,
    full_low=23.0,
    full_high=238.0,
    max_gain=TAB3_TOF_MAX_GAIN,
):
    """
    First-order ToF count multiplier from acquired mass-window span.

        raw multiplier = full mass span / selected mass span

    The result is capped at ``max_gain``. This keeps the width-based model
    consistent with the approximately fivefold maximum gain demonstrated for
    restricted-range Vitesse acquisition, instead of presenting the idealized
    span ratio as an unlimited physical sensitivity law.
    """
    full_span = float(full_high) - float(full_low)
    selected_span = float(high_mass) - float(low_mass)

    if full_span <= 0 or selected_span <= 0:
        return np.nan

    raw_gain = full_span / selected_span
    return min(float(raw_gain), max(float(max_gain), 1.0))


def first_rising_crossover(x, y_curve, horizontal_y):
    """Return the first x where an increasing curve crosses a horizontal line."""
    x = np.asarray(x, dtype=float)
    y_curve = np.asarray(y_curve, dtype=float)
    good = np.isfinite(x) & np.isfinite(y_curve)
    x = x[good]
    y_curve = y_curve[good]

    if x.size < 2 or not np.isfinite(horizontal_y):
        return np.nan

    d = y_curve - float(horizontal_y)

    for i in range(x.size - 1):
        if d[i] <= 0 <= d[i + 1]:
            if d[i + 1] == d[i]:
                return float(x[i])
            f = -d[i] / (d[i + 1] - d[i])
            return float(x[i] + f * (x[i + 1] - x[i]))

    return np.nan


def clamp_window(start_req, end_req, n_total, min_n=3):
    if n_total < min_n:
        return None, None

    start = max(0, int(start_req))
    end = min(n_total, max(start + 1, int(end_req)))

    if end - start < min_n:
        start = max(0, end - min_n)
        end = min(n_total, start + min_n)

    if end - start < min_n:
        return None, None

    return start, end


def analyte_pixel_rsd(signal_means, total_vars):
    """
    Element-specific image-pixel RSD.

    For one analyte/element, ``signal_means`` contains the expected analyte
    counts in each image pixel. ``total_vars`` contains the propagated Poisson
    variance from signal plus background for those same pixels.

    The reported value is therefore the RSD of one element across image pixels,
    not an RSD calculated among different elements and not a per-shot RSD.

    The denominator is SIGNAL ONLY. This prevents the fake low-count plateau
    that happens when signal + background is used as the denominator.
    """
    signal_means = np.asarray(signal_means, dtype=float)
    total_vars = np.asarray(total_vars, dtype=float)

    good = np.isfinite(signal_means) & np.isfinite(total_vars)
    signal_means = signal_means[good]
    total_vars = total_vars[good]

    if signal_means.size < 2:
        return np.nan, np.nan, np.nan, np.nan

    mean_signal = float(np.mean(signal_means))

    if mean_signal <= 0:
        return mean_signal, np.nan, np.nan, np.nan

    deterministic_var = float(np.var(signal_means, ddof=1))
    poisson_var = float(np.mean(np.maximum(total_vars, 0.0)))
    total_var = deterministic_var + poisson_var

    rsd = 100.0 * np.sqrt(total_var) / mean_signal

    return mean_signal, rsd, deterministic_var, poisson_var


# ============================================================
# CONCENTRATION / SIGNAL HELPERS
# ============================================================

def spot_area_um2_from_diameter(spot_um):
    r = max(float(spot_um), 0.0) / 2.0
    return np.pi * r * r


def shot_volume_um3(spot_um, depth_um):
    return spot_area_um2_from_diameter(spot_um) * max(float(depth_um), 0.0)


def sensitivity_constant_from_anchor(ref_cps, ref_conc_ppm, ref_rep_hz, ref_spot_um, ref_depth_um):
    """
    Anchor model:
        ref_cps = K * concentration_ppm * volume_per_shot_um3 * rep_rate_hz
    """
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


def counts_per_shot_from_anchor_and_class(
    class_name,
    ultra_trace_ppm,
    trace_ppm,
    minor_ppm,
    major_ppm,
    ref_cps,
    ref_conc_ppm,
    ref_rep_hz,
    ref_spot_um,
    ref_depth_um,
    spot_um,
    depth_um,
):
    conc_ppm = class_concentration_map(
        ultra_trace_ppm,
        trace_ppm,
        minor_ppm,
        major_ppm,
    )[class_name]

    k = sensitivity_constant_from_anchor(
        ref_cps=ref_cps,
        ref_conc_ppm=ref_conc_ppm,
        ref_rep_hz=ref_rep_hz,
        ref_spot_um=ref_spot_um,
        ref_depth_um=ref_depth_um,
    )

    return k * conc_ppm * shot_volume_um3(spot_um, depth_um)


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
    """
    Same pulse shape logic as the ToF simulator:
        finite rise + fast exponential tail + optional slow exponential tail.
    """
    washout_ms = max(float(washout_ms), 1e-6)

    rise_frac, slow_frac, slow_tau_mult = shape_params(shape_name)

    tau_fast_s = max((washout_ms / 6.0) / 1000.0, dt_sec)
    tau_slow_s = max(tau_fast_s * slow_tau_mult, dt_sec)
    tau_rise_s = max((washout_ms * rise_frac) / 1000.0, dt_sec)

    kernel_duration_s = max(6.0 * tau_slow_s, 4.0 * washout_ms / 1000.0)
    t_kernel = np.arange(0.0, kernel_duration_s + dt_sec, dt_sec)

    rise_term = 1.0 - np.exp(-t_kernel / tau_rise_s)

    tail = (
        (1.0 - slow_frac) * np.exp(-t_kernel / tau_fast_s)
        + slow_frac * np.exp(-t_kernel / tau_slow_s)
    )

    kernel = np.maximum(rise_term * tail, 0.0)

    if np.all(kernel <= 0):
        kernel = np.zeros_like(t_kernel)
        kernel[0] = 1.0

    return t_kernel, kernel


# ============================================================
# SAMPLING HELPERS
# ============================================================

def interval_sums_from_rate(rate_cps, dt_s, starts_s, width_s):
    """
    Integrate a high-resolution cps signal over arbitrary time windows.
    """
    rate_cps = np.asarray(rate_cps, dtype=float)
    starts_s = np.asarray(starts_s, dtype=float)

    if rate_cps.size == 0 or starts_s.size == 0 or dt_s <= 0 or width_s <= 0:
        return np.array([], dtype=float)

    n = rate_cps.size
    edges = np.arange(n + 1, dtype=float) * dt_s
    cumulative = np.r_[0.0, np.cumsum(rate_cps) * dt_s]

    out = np.zeros(starts_s.size, dtype=float)

    for i, t0_raw in enumerate(starts_s):
        t0 = max(float(t0_raw), edges[0])
        t1 = min(float(t0_raw) + width_s, edges[-1])

        if t1 <= t0:
            continue

        c0 = np.interp(t0, edges, cumulative)
        c1 = np.interp(t1, edges, cumulative)
        out[i] = c1 - c0

    return np.maximum(out, 0.0)


def interval_sums_from_rate_fast(rate_cps, dt_s, starts_s, width_s):
    """
    Vectorized equivalent of ``interval_sums_from_rate``.

    This is used by Tab 3 because the QMS fields evaluate several sweep,
    switching, sensitivity, and phase scenarios. It integrates the same
    high-resolution laser-ablation signal; it only avoids a Python loop over
    every dwell window.
    """
    rate_cps = np.asarray(rate_cps, dtype=float)
    starts_s = np.asarray(starts_s, dtype=float)

    if rate_cps.size == 0 or starts_s.size == 0 or dt_s <= 0 or width_s <= 0:
        return np.array([], dtype=float)

    duration_s = rate_cps.size * float(dt_s)
    starts = np.clip(starts_s, 0.0, duration_s)
    ends = np.clip(starts_s + float(width_s), 0.0, duration_s)

    edges = np.arange(rate_cps.size + 1, dtype=float) * float(dt_s)
    cumulative = np.r_[0.0, np.cumsum(rate_cps) * float(dt_s)]

    c0 = np.interp(starts, edges, cumulative)
    c1 = np.interp(ends, edges, cumulative)

    out = c1 - c0
    out[ends <= starts] = 0.0
    return np.maximum(out, 0.0)


def qms_phase_windows(duration_s, sweep_s, dwell_s, phase_s):
    """
    Return repeated QMS dwell windows for one mass at a specified cycle phase.

    ``phase_s`` is the start of that mass's dwell within the QMS cycle. Sampling
    several phases represents different element positions and laser/QMS phase
    relationships without changing the physical laser-ablation signal.
    """
    duration_s = float(duration_s)
    sweep_s = float(sweep_s)
    dwell_s = float(dwell_s)

    if duration_s <= 0 or sweep_s <= 0 or dwell_s <= 0:
        return np.array([], dtype=float)

    phase_s = float(phase_s) % sweep_s
    k0 = int(np.floor((-phase_s - dwell_s) / sweep_s))
    k1 = int(np.ceil((duration_s - phase_s) / sweep_s))

    starts = phase_s + np.arange(k0, k1 + 1, dtype=float) * sweep_s
    keep = (starts < duration_s) & ((starts + dwell_s) > 0.0)
    return starts[keep]


def event_counts_to_window_mean_var(event_starts, event_width, event_mu_counts, window_starts, window_width):
    """
    Propagate event counts into target windows by overlap.

    Events can be ToF bins or QMS dwells.
    Target windows are pixels.

    If a pixel receives fraction w of an event:
        mean += w * mu
        variance += w^2 * mu
    """
    event_starts = np.asarray(event_starts, dtype=float)
    event_mu_counts = np.asarray(event_mu_counts, dtype=float)
    window_starts = np.asarray(window_starts, dtype=float)

    n_win = window_starts.size

    means = np.zeros(n_win, dtype=float)
    variances = np.zeros(n_win, dtype=float)

    if event_starts.size == 0 or event_mu_counts.size == 0 or n_win == 0:
        return means, variances

    if event_width <= 0 or window_width <= 0:
        return means, variances

    w0_all = float(window_starts[0])
    w_last = float(window_starts[-1] + window_width)

    for e0, mu in zip(event_starts, event_mu_counts):
        if not np.isfinite(mu):
            continue

        e1 = e0 + event_width

        if e1 <= w0_all or e0 >= w_last:
            continue

        j0 = int(np.floor((e0 - w0_all) / window_width)) - 1
        j1 = int(np.floor((e1 - w0_all) / window_width)) + 1

        j0 = max(0, j0)
        j1 = min(n_win - 1, j1)

        for j in range(j0, j1 + 1):
            w0 = window_starts[j]
            w1 = w0 + window_width

            overlap = max(0.0, min(e1, w1) - max(e0, w0))

            if overlap <= 0:
                continue

            frac = overlap / event_width

            means[j] += frac * mu
            variances[j] += frac * frac * mu

    return means, variances


def tof_events(
    rate_cps,
    dt_s,
    duration_s,
    tint_s,
    background_cps,
    signal_multiplier=1.0,
    background_multiplier=1.0,
):
    """
    ToF sampling.

    Every ToF integration bin sees all selected masses concurrently.
    signal_multiplier is used for the restricted range / 2x count case.
    """
    tint_s = float(tint_s)

    if tint_s <= 0 or len(rate_cps) == 0:
        return np.array([]), np.array([]), np.nan

    starts = np.arange(0.0, duration_s, tint_s)

    mu_signal = interval_sums_from_rate(
        rate_cps * float(signal_multiplier),
        dt_s,
        starts,
        tint_s,
    )

    mu_bkg = max(0.0, float(background_cps)) * float(background_multiplier) * tint_s
    mu_total = mu_signal + mu_bkg

    return starts, np.maximum(mu_total, 0.0), tint_s


# ============================================================
# QMS HELPERS
# ============================================================

def quad_dwell_time_per_element(sweep_s, switch_s, n_elements):
    """
    Equal-dwell quadrupole sweep model with N switch/dead-time losses per cycle:

        dwell = (sweep - N * switch) / N

    This includes the switch back to the beginning of the next sweep.
    """
    n_elements = int(n_elements)

    if n_elements <= 0:
        return np.nan

    dwell_s = (float(sweep_s) - n_elements * float(switch_s)) / n_elements

    if dwell_s <= 0:
        return np.nan

    return dwell_s


def quad_element_windows(duration_s, sweep_s, switch_s, n_elements, element_index):
    dwell_s = quad_dwell_time_per_element(sweep_s, switch_s, n_elements)

    if not np.isfinite(dwell_s):
        return np.array([]), np.nan

    element_index = int(element_index)

    if element_index < 0 or element_index >= int(n_elements):
        return np.array([]), np.nan

    sweep_starts = np.arange(0.0, duration_s, sweep_s)

    offset = switch_s + element_index * (dwell_s + switch_s)

    starts = sweep_starts + offset
    starts = starts[starts < duration_s]

    return starts, dwell_s


def qms_events(
    rate_cps_tof_sensitivity,
    dt_s,
    duration_s,
    n_elements,
    sweep_s,
    switch_s,
    qms_sensitivity_mult,
    element_index,
    background_cps,
):
    """
    QMS sampling.

    Same LA signal, but one element is integrated only during its dwell windows.

    Signal and background both scale by the QMS sensitivity multiplier.
    """
    starts, dwell_s = quad_element_windows(
        duration_s=duration_s,
        sweep_s=sweep_s,
        switch_s=switch_s,
        n_elements=n_elements,
        element_index=element_index,
    )

    if not np.isfinite(dwell_s) or starts.size == 0:
        return np.array([]), np.array([]), np.nan

    mult = float(qms_sensitivity_mult)

    mu_signal = interval_sums_from_rate(
        rate_cps_tof_sensitivity * mult,
        dt_s,
        starts,
        dwell_s,
    )

    mu_bkg = max(0.0, float(background_cps)) * mult * dwell_s
    mu_total = mu_signal + mu_bkg

    return starts, np.maximum(mu_total, 0.0), dwell_s


# ============================================================
# APPLICATION
# ============================================================

class QuadVsToFApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quad vs ToF Sampling and Precision Model")
        self.root.geometry("1580x1000")

        self.export_data = {}

        self._set_vars()
        self._build_ui()
        self.update_all()

    def _set_vars(self):
        self.freq_var = tk.DoubleVar(value=500.0)
        self.washout_ms_var = tk.DoubleVar(value=2.0)
        self.duration_s_var = tk.DoubleVar(value=1.0)

        self.spot_um_var = tk.DoubleVar(value=5.0)
        self.depth_per_shot_um_var = tk.DoubleVar(value=0.1)

        self.dose_var = tk.IntVar(value=5)
        self.start_pix_var = tk.IntVar(value=10)
        self.end_pix_var = tk.IntVar(value=200)

        self.tof_display_tint_ms_var = tk.DoubleVar(value=1.0)
        self.qms_display_elements_var = tk.IntVar(value=5)
        self.qms_display_element_counts_var = tk.IntVar(value=1)

        self.quad_sweep_ms_var = tk.DoubleVar(value=5.0)
        self.quad_switch_ms_var = tk.DoubleVar(value=0.2)
        self.quad_sensitivity_mult_var = tk.DoubleVar(value=10.0)
        self.max_elements_var = tk.IntVar(value=20)
        self.tof_restricted_mult_var = tk.DoubleVar(value=2.0)

        # Tab 3 uses three explicit sweep conditions. Switching/settling time
        # comes from the current Quad switch control; the line sensitivity comes
        # from the current Quad sensitivity control.
        self.tab3_qms_sweeps_var = tk.StringVar(value=TAB3_DEFAULT_QMS_SWEEPS_MS)
        self.tab3_qms_sensitivities_var = tk.StringVar(value=TAB3_DEFAULT_QMS_SENSITIVITIES)
        self.tab3_min_dwell_ms_var = tk.DoubleVar(value=TAB3_DEFAULT_MIN_DWELL_MS)

        self.ref_conc_ppm_var = tk.DoubleVar(value=500.0)
        self.ref_cps_var = tk.DoubleVar(value=300000.0)
        self.ref_rep_hz_var = tk.DoubleVar(value=200.0)
        self.ref_spot_um_var = tk.DoubleVar(value=5.0)
        self.ref_depth_um_var = tk.DoubleVar(value=0.1)

        self.ultra_trace_ppm_var = tk.DoubleVar(value=0.1)
        self.trace_ppm_var = tk.DoubleVar(value=10.0)
        self.minor_ppm_var = tk.DoubleVar(value=1000.0)
        self.major_ppm_var = tk.DoubleVar(value=10000.0)
        self.active_class_var = tk.StringVar(value="Trace")

        self.signal_mode_var = tk.StringVar(value="Class-based")
        self.manual_counts_per_shot_var = tk.DoubleVar(value=10.0)

        self.shape_profile_var = tk.StringVar(value="broad")

        self.shot_yield_rsd_var = tk.DoubleVar(value=5.0)
        self.transport_rsd_var = tk.DoubleVar(value=0.0)
        self.shot_jitter_us_var = tk.DoubleVar(value=0.0)

        self.background_cps_var = tk.DoubleVar(value=10.0)

        self.zoom_start_s_var = tk.DoubleVar(value=0.04)
        self.zoom_end_s_var = tk.DoubleVar(value=0.20)

        self.show_poisson_var = tk.IntVar(value=1)

        self.stat_counts_per_shot = tk.StringVar(value="--")
        self.stat_signal_cps = tk.StringVar(value="--")
        self.stat_pixel_time = tk.StringVar(value="--")
        self.stat_qms_n1 = tk.StringVar(value="--")
        self.stat_qms_n5 = tk.StringVar(value="--")
        self.stat_qms_n20 = tk.StringVar(value="--")

    def _build_ui(self):
        top = tk.Frame(self.root, bg="#f0f0f0")
        top.pack(side=tk.TOP, fill=tk.X)

        control = ttk.LabelFrame(top, text="Parameters")
        control.pack(side=tk.TOP, fill=tk.X, padx=8, pady=5)

        r = 0
        c = 0

        self.create_slider(control, "Laser Freq (Hz)", self.freq_var, 1, 1000, r, c); c += 1
        self.create_slider(control, "Washout (ms)", self.washout_ms_var, 0.5, 10.0, r, c); c += 1
        self.create_slider(control, "Duration (s)", self.duration_s_var, 0.2, 5.0, r, c); c += 1
        self.create_slider(control, "Spot (µm)", self.spot_um_var, 1.0, 50.0, r, c); c += 1
        self.create_slider(control, "Depth/shot (µm)", self.depth_per_shot_um_var, 0.01, 1.0, r, c); c += 1
        self.create_slider(control, "Dose shots/pix", self.dose_var, 1, 50, r, c); c += 1
        self.create_entry_int(control, "Start pix", self.start_pix_var, r, c); c += 1
        self.create_entry_int(control, "End pix", self.end_pix_var, r, c); c += 1

        class_frame = ttk.Frame(control)
        class_frame.grid(row=r, column=c, padx=6, pady=2, sticky="ew")

        ttk.Label(class_frame, text="Class:", font=("Arial", 8)).pack(side=tk.LEFT)

        ttk.Combobox(
            class_frame,
            textvariable=self.active_class_var,
            values=CONC_CLASSES,
            state="readonly",
            width=12,
        ).pack(side=tk.LEFT, padx=(6, 0))

        c += 1

        r = 1
        c = 0

        self.create_entry(control, "ToF display Tint ms", self.tof_display_tint_ms_var, r, c); c += 1
        self.create_entry_int(control, "Display QMS elems", self.qms_display_elements_var, r, c); c += 1
        self.create_entry_int(control, "Bottom QMS elem #", self.qms_display_element_counts_var, r, c); c += 1
        self.create_slider(control, "Quad sweep (ms)", self.quad_sweep_ms_var, 0.5, 1000.0, r, c); c += 1
        self.create_slider(control, "Quad switch (ms)", self.quad_switch_ms_var, 0.0, 50.0, r, c); c += 1
        self.create_slider(control, "Quad sensitivity ×", self.quad_sensitivity_mult_var, 0.1, 50.0, r, c); c += 1
        self.create_entry_int(control, "Max elements", self.max_elements_var, r, c); c += 1
        self.create_entry(control, "Tab 4 restricted ToF ×", self.tof_restricted_mult_var, r, c); c += 1
        self.create_slider(control, "Background cps", self.background_cps_var, 0.0, 5000.0, r, c); c += 1

        r = 2
        c = 0

        self.create_slider(control, "Shot yield RSD (%)", self.shot_yield_rsd_var, 0.0, 50.0, r, c); c += 1
        self.create_slider(control, "Transport RSD (%)", self.transport_rsd_var, 0.0, 50.0, r, c); c += 1
        self.create_slider(control, "Shot jitter (µs)", self.shot_jitter_us_var, 0.0, 200.0, r, c); c += 1
        self.create_entry(control, "Zoom start (s)", self.zoom_start_s_var, r, c); c += 1
        self.create_entry(control, "Zoom end (s)", self.zoom_end_s_var, r, c); c += 1

        shape_frame = ttk.Frame(control)
        shape_frame.grid(row=r, column=c, padx=6, pady=2, sticky="ew")

        ttk.Label(shape_frame, text="Shape:", font=("Arial", 8)).pack(side=tk.LEFT)

        for shape in SHAPE_ORDER:
            ttk.Radiobutton(
                shape_frame,
                text=shape,
                variable=self.shape_profile_var,
                value=shape,
            ).pack(side=tk.LEFT, padx=(4, 0))

        c += 1

        mode_frame = ttk.Frame(control)
        mode_frame.grid(row=r, column=c, padx=6, pady=2, sticky="ew")

        ttk.Label(mode_frame, text="Signal mode:", font=("Arial", 8)).pack(side=tk.LEFT)

        ttk.Radiobutton(
            mode_frame,
            text="Class-based",
            variable=self.signal_mode_var,
            value="Class-based",
        ).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Radiobutton(
            mode_frame,
            text="Manual cts/shot",
            variable=self.signal_mode_var,
            value="Manual cts/shot",
        ).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Entry(
            mode_frame,
            textvariable=self.manual_counts_per_shot_var,
            width=8,
        ).pack(side=tk.LEFT, padx=(6, 0))

        c += 1

        ttk.Checkbutton(
            control,
            text="Show Poisson example",
            variable=self.show_poisson_var,
        ).grid(row=r, column=c, padx=6, pady=2)

        r = 3
        c = 0

        self.create_entry(control, "Ref conc ppm", self.ref_conc_ppm_var, r, c); c += 1
        self.create_entry(control, "Ref signal cps", self.ref_cps_var, r, c); c += 1
        self.create_entry(control, "Ref rep Hz", self.ref_rep_hz_var, r, c); c += 1
        self.create_entry(control, "Ref spot µm", self.ref_spot_um_var, r, c); c += 1
        self.create_entry(control, "Ref depth µm", self.ref_depth_um_var, r, c); c += 1

        r = 4
        c = 0

        self.create_entry(control, "Ultra trace ppm", self.ultra_trace_ppm_var, r, c); c += 1
        self.create_entry(control, "Trace ppm", self.trace_ppm_var, r, c); c += 1
        self.create_entry(control, "Minor ppm", self.minor_ppm_var, r, c); c += 1
        self.create_entry(control, "Major ppm", self.major_ppm_var, r, c); c += 1
        self.create_entry_wide(control, "Tab 3 QMS sweeps (ms)", self.tab3_qms_sweeps_var, r, c); c += 1
        self.create_entry_wide(control, "Tab 3 QMS sensitivity bounds ×", self.tab3_qms_sensitivities_var, r, c); c += 1
        self.create_entry(control, "Tab 3 minimum dwell (ms)", self.tab3_min_dwell_ms_var, r, c); c += 1

        for i in range(16):
            control.columnconfigure(i, weight=1)

        btns = tk.Frame(top, bg="#f0f0f0")
        btns.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        tk.Button(
            btns,
            text="UPDATE",
            command=self.update_all,
            bg="#e6f2ff",
            font=("Arial", 10, "bold"),
            width=18,
        ).pack(side=tk.LEFT, padx=(4, 10))

        tk.Button(
            btns,
            text="EXPORT CURRENT TAB CSV",
            command=self.export_current_tab_csv,
            bg="#f5f5f5",
            font=("Arial", 10, "bold"),
            width=24,
        ).pack(side=tk.LEFT, padx=(4, 10))

        tk.Button(
            btns,
            text="EXPORT CURRENT TAB PDF",
            command=self.export_current_tab_pdf,
            bg="#f5f5f5",
            font=("Arial", 10, "bold"),
            width=24,
        ).pack(side=tk.LEFT, padx=(4, 10))

        tk.Label(
            btns,
            text="Core model: identical LA pulses → washout → ToF integration or QMS dwell sampling.",
            bg="#f0f0f0",
            fg="#8a3b00",
            font=("Arial", 9, "bold"),
        ).pack(side=tk.LEFT)

        stats = tk.Frame(self.root, bg="#e1e4e8", height=55)
        stats.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        self.make_stat(stats, "ToF counts/shot", self.stat_counts_per_shot)
        self.make_stat(stats, "ToF signal cps", self.stat_signal_cps)
        self.make_stat(stats, "Pixel time", self.stat_pixel_time)
        self.make_stat(stats, "QMS N=1 dwell", self.stat_qms_n1)
        self.make_stat(stats, "QMS N=5 dwell", self.stat_qms_n5)
        self.make_stat(stats, "QMS N=20 dwell", self.stat_qms_n20)

        plot_container = tk.Frame(self.root, bg="white", bd=2, relief=tk.SUNKEN)
        plot_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.notebook = ttk.Notebook(plot_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_signal = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_signal, text="1. Raw LA Signal")
        self.fig_signal, self.ax_signal, self.canvas_signal = self.make_plot_tab(self.tab_signal)

        self.tab_sampling = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_sampling, text="2. Sampling Geometry")
        self.fig_sampling, self.ax_sampling, self.canvas_sampling = self.make_plot_tab(self.tab_sampling)

        self.tab_shot = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_shot, text="3. Element-Specific Pixel RSD")
        self.fig_shot, self.ax_shot, self.canvas_shot = self.make_plot_tab(self.tab_shot)

        self.tab_pixel = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_pixel, text="4. Pixel-Pixel RSD")
        self.fig_pixel, self.ax_pixel, self.canvas_pixel = self.make_plot_tab(self.tab_pixel)

    def make_stat(self, parent, label, var):
        f = tk.Frame(parent, bg="#e1e4e8", bd=1, relief=tk.RIDGE)
        f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)

        tk.Label(
            f,
            text=label,
            bg="#e1e4e8",
            fg="#555",
            font=("Arial", 8, "bold"),
        ).pack(pady=1)

        tk.Label(
            f,
            textvariable=var,
            bg="#e1e4e8",
            fg="#007acc",
            font=("Arial", 11, "bold"),
        ).pack(pady=1)

    def make_plot_tab(self, parent):
        frame = tk.Frame(parent, bg="white")
        frame.pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(frame, bg="#cccccc")
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        fig, ax = plt.subplots(figsize=(11.8, 6.8))

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        return fig, ax, canvas

    def make_two_plot_tab(self, parent):
        frame = tk.Frame(parent, bg="white")
        frame.pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(frame, bg="#cccccc")
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.8), sharey=False)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        return fig, axes, canvas

    def create_slider(self, parent, label, variable, min_val, max_val, row, col):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, padx=6, pady=2, sticky="ew")

        ttk.Label(frame, text=label, font=("Arial", 8)).grid(row=0, column=0, sticky="w")

        ttk.Scale(
            frame,
            from_=min_val,
            to=max_val,
            variable=variable,
            orient=tk.HORIZONTAL,
        ).grid(row=0, column=1, sticky="ew", padx=(6, 6))

        ttk.Entry(frame, textvariable=variable, width=8).grid(row=0, column=2, sticky="e")

        frame.columnconfigure(1, weight=1)

    def create_entry_int(self, parent, label, variable, row, col):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, padx=6, pady=2, sticky="ew")

        ttk.Label(frame, text=label, font=("Arial", 8)).grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=variable, width=8).grid(row=0, column=1, sticky="e", padx=(6, 0))

        frame.columnconfigure(0, weight=1)

    def create_entry(self, parent, label, variable, row, col):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, padx=6, pady=2, sticky="ew")

        ttk.Label(frame, text=label, font=("Arial", 8)).grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=variable, width=8).grid(row=0, column=1, sticky="e", padx=(6, 0))

        frame.columnconfigure(0, weight=1)

    def create_entry_wide(self, parent, label, variable, row, col):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, padx=6, pady=2, sticky="ew")

        ttk.Label(frame, text=label, font=("Arial", 8)).grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=variable, width=16).grid(row=0, column=1, sticky="e", padx=(6, 0))

        frame.columnconfigure(0, weight=1)

    # --------------------------------------------------------
    # MODEL STATE
    # --------------------------------------------------------

    def current_counts_per_shot(self, class_name=None):
        if class_name is None:
            class_name = self.active_class_var.get()

        if str(self.signal_mode_var.get()) == "Manual cts/shot":
            return max(0.0, float(self.manual_counts_per_shot_var.get()))

        return counts_per_shot_from_anchor_and_class(
            class_name=class_name,
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

    def current_signal_cps(self, class_name=None, rep_hz=None):
        if class_name is None:
            class_name = self.active_class_var.get()

        if rep_hz is None:
            rep_hz = float(self.freq_var.get())

        return self.current_counts_per_shot(class_name) * max(float(rep_hz), 0.0)

    def scenario_seed(self, base, *vals):
        out = int(base)

        for i, v in enumerate(vals):
            try:
                vv = float(v)
            except Exception:
                vv = float(abs(hash(str(v))) % 100000)

            out = (
                out * 1664525
                + 1013904223
                + int(round(vv * (11 + 2 * i)))
            ) % (2**32 - 1)

        return int(out)

    def ideal_signal_highres(self, class_name=None, rep_hz=None, rng=None):
        if rng is None:
            rng = np.random.default_rng(1)

        if class_name is None:
            class_name = self.active_class_var.get()

        if rep_hz is None:
            rep_hz = float(self.freq_var.get())

        duration = float(self.duration_s_var.get())
        washout_ms = float(self.washout_ms_var.get())
        shape_name = str(self.shape_profile_var.get()).strip().lower()
        counts_per_shot_mean = self.current_counts_per_shot(class_name=class_name)

        dt_s = DT_US / 1e6
        time_s = np.arange(0.0, duration, dt_s)

        if time_s.size == 0:
            return np.array([]), np.array([]), dt_s

        shot_period_s = 1.0 / max(float(rep_hz), 1e-12)
        shot_times = np.arange(0.0, duration, shot_period_s)
        n_shots = shot_times.size

        shot_factors = (
            sample_positive_factors(self.shot_yield_rsd_var.get(), n_shots, rng)
            * sample_positive_factors(self.transport_rsd_var.get(), n_shots, rng)
        )

        jitter_s = max(0.0, float(self.shot_jitter_us_var.get())) * 1e-6

        if jitter_s > 0:
            shot_times = shot_times + rng.normal(0.0, jitter_s, size=n_shots)

        shot_times = np.clip(shot_times, 0.0, max(duration - dt_s, 0.0))
        shot_idx = np.clip((shot_times / dt_s).astype(int), 0, time_s.size - 1)

        triggers = np.zeros(time_s.size, dtype=float)
        np.add.at(triggers, shot_idx, counts_per_shot_mean * shot_factors)

        _, kernel = washout_kernel_same_model(washout_ms, dt_s, shape_name)

        kernel_area = float(np.sum(kernel) * dt_s)
        if kernel_area <= 0:
            kernel_area = 1.0

        ideal_rate_cps = convolve(triggers, kernel, mode="full")[:time_s.size] / kernel_area

        return time_s, np.maximum(ideal_rate_cps, 0.0), dt_s

    def ensure_signal(self):
        self.time_s, self.ideal_rate_cps, self.dt_s = self.ideal_signal_highres(
            class_name=self.active_class_var.get(),
            rep_hz=float(self.freq_var.get()),
            rng=np.random.default_rng(
                self.scenario_seed(
                    1001,
                    self.freq_var.get(),
                    self.washout_ms_var.get(),
                    self.current_counts_per_shot(),
                    self.shot_yield_rsd_var.get(),
                    self.transport_rsd_var.get(),
                    self.shape_profile_var.get(),
                )
            ),
        )

    def pixel_window(self):
        rep_hz = float(self.freq_var.get())
        dose = int(self.dose_var.get())
        duration_s = float(self.duration_s_var.get())

        pixel_width_s = dose / max(rep_hz, 1e-12)
        n_pix_total = int(np.floor(duration_s / pixel_width_s))

        sp, ep = clamp_window(
            self.start_pix_var.get(),
            self.end_pix_var.get(),
            n_pix_total,
            min_n=3,
        )

        if sp is None:
            return None, None, None, None, None

        pixel_indices = np.arange(sp, ep, dtype=int)
        pixel_starts_s = pixel_indices * pixel_width_s

        return pixel_indices, pixel_starts_s, pixel_width_s, sp, ep

    def scaled_rate_for_counts_per_pulse(self, counts_per_pulse):
        current = self.current_counts_per_shot(self.active_class_var.get())

        if not finite_positive(current):
            return None

        return self.ideal_rate_cps * (float(counts_per_pulse) / current)

    def update_all(self):
        try:
            self.ensure_signal()
            self.update_stats()

            tab = self.notebook.tab(self.notebook.select(), "text")

            if tab.startswith("1."):
                self.update_raw_signal_tab()
            elif tab.startswith("2."):
                self.update_sampling_tab()
            elif tab.startswith("3."):
                self.update_shot_rsd_tab()
            elif tab.startswith("4."):
                self.update_pixel_rsd_tab()
            else:
                self.update_raw_signal_tab()

        except Exception as e:
            messagebox.showerror("Model error", str(e))

    def update_stats(self):
        class_name = self.active_class_var.get()

        cpshot = self.current_counts_per_shot(class_name)
        cps = self.current_signal_cps(class_name)

        pixel_time_s = int(self.dose_var.get()) / max(float(self.freq_var.get()), 1e-12)

        sweep_s = float(self.quad_sweep_ms_var.get()) / 1000.0
        switch_s = float(self.quad_switch_ms_var.get()) / 1000.0

        def dwell_text(n):
            dwell_s = quad_dwell_time_per_element(sweep_s, switch_s, n)
            return f"{dwell_s * 1e3:.3g} ms" if np.isfinite(dwell_s) else "invalid"

        self.stat_counts_per_shot.set(f"{cpshot:.3g}")
        self.stat_signal_cps.set(f"{cps:,.0f}")
        self.stat_pixel_time.set(f"{pixel_time_s * 1e3:.3g} ms")
        self.stat_qms_n1.set(dwell_text(1))
        self.stat_qms_n5.set(dwell_text(5))
        self.stat_qms_n20.set(dwell_text(20))

    # --------------------------------------------------------
    # TAB 1
    # --------------------------------------------------------

    def update_raw_signal_tab(self):
        ax = self.ax_signal
        ax.clear()

        duration_s = float(self.duration_s_var.get())

        zoom_start = max(0.0, float(self.zoom_start_s_var.get()))
        zoom_end = min(duration_s, float(self.zoom_end_s_var.get()))

        if zoom_end <= zoom_start:
            zoom_start = 0.0
            zoom_end = min(duration_s, 0.2)

        mask = (self.time_s >= zoom_start) & (self.time_s <= zoom_end)

        ax.plot(
            self.time_s[mask],
            self.ideal_rate_cps[mask],
            color="black",
            linewidth=1.2,
        )

        rep_hz = float(self.freq_var.get())
        shot_period = 1.0 / max(rep_hz, 1e-12)
        shot_times = np.arange(0.0, duration_s, shot_period)

        for st in shot_times:
            if zoom_start <= st <= zoom_end:
                ax.axvline(st, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)

        class_name = self.active_class_var.get()

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("LA signal rate (ToF-equivalent cps)")
        ax.set_title(
            f"Raw LA signal | {class_name} | "
            f"{self.current_counts_per_shot(class_name):.3g} ToF counts/shot | "
            f"washout={self.washout_ms_var.get():g} ms | "
            f"shape={self.shape_profile_var.get()}"
        )
        ax.grid(True, alpha=0.25)
        ax.set_xlim(zoom_start, zoom_end)

        rows = []

        for t, y in zip(self.time_s[mask], self.ideal_rate_cps[mask]):
            rows.append({
                "panel": "raw_la_signal",
                "time_s": t,
                "rate_cps_tof_equivalent": y,
                "class": class_name,
                "counts_per_shot_tof_equivalent": self.current_counts_per_shot(class_name),
                "rep_rate_hz": rep_hz,
                "washout_ms": self.washout_ms_var.get(),
                "shape": self.shape_profile_var.get(),
            })

        self.export_data["1. Raw LA Signal"] = rows

        self.fig_signal.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.12)
        self.canvas_signal.draw()

    # --------------------------------------------------------
    # TAB 2
    # --------------------------------------------------------

    def update_sampling_tab(self):
        self.fig_sampling.clf()

        gs = self.fig_sampling.add_gridspec(
            2,
            1,
            height_ratios=[1.15, 1.0],
            hspace=0.28,
        )

        ax_top = self.fig_sampling.add_subplot(gs[0, 0])
        ax_bot = self.fig_sampling.add_subplot(gs[1, 0], sharex=ax_top)

        duration_s = float(self.duration_s_var.get())

        zoom_start = max(0.0, float(self.zoom_start_s_var.get()))
        zoom_end = min(duration_s, float(self.zoom_end_s_var.get()))

        if zoom_end <= zoom_start:
            zoom_start = 0.0
            zoom_end = min(duration_s, 0.2)

        mask = (self.time_s >= zoom_start) & (self.time_s <= zoom_end)

        tof_tint_s = max(float(self.tof_display_tint_ms_var.get()) / 1000.0, self.dt_s)

        display_n = max(1, int(self.qms_display_elements_var.get()))
        count_element = max(
            0,
            min(display_n - 1, int(self.qms_display_element_counts_var.get()) - 1),
        )

        sweep_s = float(self.quad_sweep_ms_var.get()) / 1000.0
        switch_s = float(self.quad_switch_ms_var.get()) / 1000.0
        qmult = float(self.quad_sensitivity_mult_var.get())

        ax_top.plot(
            self.time_s[mask],
            self.ideal_rate_cps[mask],
            color="black",
            linewidth=1.0,
            label="Raw LA signal",
        )

        tof_starts, tof_mu, _ = tof_events(
            self.ideal_rate_cps,
            self.dt_s,
            duration_s,
            tof_tint_s,
            background_cps=float(self.background_cps_var.get()),
            signal_multiplier=1.0,
            background_multiplier=1.0,
        )

        tof_centers = tof_starts + 0.5 * tof_tint_s
        tof_in = (tof_centers >= zoom_start) & (tof_centers <= zoom_end)

        rows = []

        for t, y in zip(self.time_s[mask], self.ideal_rate_cps[mask]):
            rows.append({
                "panel": "raw_signal",
                "series": "raw_la_signal",
                "time_s": t,
                "value": y,
                "units": "cps",
            })

        if np.any(tof_in):
            tof_rate = tof_mu / tof_tint_s

            ax_top.plot(
                tof_centers[tof_in],
                tof_rate[tof_in],
                color="C1",
                marker="o",
                linestyle="none",
                markersize=4,
                label=f"ToF bins, {tof_tint_s * 1e3:g} ms",
            )

            ax_bot.step(
                tof_centers[tof_in],
                tof_mu[tof_in],
                where="mid",
                color="C1",
                linewidth=2.0,
                label=f"ToF counts/bin, {tof_tint_s * 1e3:g} ms",
            )

            if self.show_poisson_var.get():
                rng = np.random.default_rng(
                    self.scenario_seed(2020, tof_tint_s * 1e6, self.background_cps_var.get())
                )

                obs = rng.poisson(tof_mu)

                ax_bot.plot(
                    tof_centers[tof_in],
                    obs[tof_in],
                    color="C1",
                    marker="o",
                    linestyle="none",
                    markersize=3,
                    alpha=0.45,
                    label="ToF Poisson example",
                )

            for t, c in zip(tof_centers[tof_in], tof_mu[tof_in]):
                rows.append({
                    "panel": "counts",
                    "series": "tof_counts_per_bin",
                    "time_s": t,
                    "value": c,
                    "units": "counts",
                    "tof_integration_ms": tof_tint_s * 1000.0,
                })

        for elem in range(display_n):
            color = QMS_OVERLAY_COLORS[elem % len(QMS_OVERLAY_COLORS)]

            starts, mu, dwell_s = qms_events(
                rate_cps_tof_sensitivity=self.ideal_rate_cps,
                dt_s=self.dt_s,
                duration_s=duration_s,
                n_elements=display_n,
                sweep_s=sweep_s,
                switch_s=switch_s,
                qms_sensitivity_mult=qmult,
                element_index=elem,
                background_cps=float(self.background_cps_var.get()),
            )

            if not np.isfinite(dwell_s) or starts.size == 0:
                continue

            centers = starts + 0.5 * dwell_s
            in_zoom = (centers >= zoom_start) & (centers <= zoom_end)

            for s0 in starts:
                if s0 > zoom_end or (s0 + dwell_s) < zoom_start:
                    continue

                ax_top.axvspan(
                    max(s0, zoom_start),
                    min(s0 + dwell_s, zoom_end),
                    color=color,
                    alpha=0.12,
                )

            if np.any(in_zoom):
                ax_top.plot(
                    centers[in_zoom],
                    mu[in_zoom] / dwell_s,
                    marker="s",
                    linestyle="none",
                    markersize=4,
                    color=color,
                    label=f"QMS elem {elem + 1}",
                )

                for t, c in zip(centers[in_zoom], mu[in_zoom]):
                    rows.append({
                        "panel": "counts",
                        "series": f"qms_element_{elem + 1}_counts_per_dwell",
                        "time_s": t,
                        "value": c,
                        "units": "counts",
                        "n_elements": display_n,
                        "element_index_1based": elem + 1,
                        "dwell_ms": dwell_s * 1000.0,
                    })

            if elem == count_element and np.any(in_zoom):
                ax_bot.plot(
                    centers[in_zoom],
                    mu[in_zoom],
                    color=color,
                    marker="s",
                    linestyle="-",
                    linewidth=1.8,
                    markersize=4,
                    label=f"QMS elem {elem + 1} counts/dwell, N={display_n}",
                )

                if self.show_poisson_var.get():
                    rng = np.random.default_rng(
                        self.scenario_seed(3030, display_n, elem, dwell_s * 1e6)
                    )

                    obs = rng.poisson(mu)

                    ax_bot.plot(
                        centers[in_zoom],
                        obs[in_zoom],
                        color=color,
                        marker="s",
                        linestyle="none",
                        markersize=3,
                        alpha=0.45,
                        label="QMS Poisson example",
                    )

        dwell_s = quad_dwell_time_per_element(sweep_s, switch_s, display_n)

        ax_top.set_ylabel("Rate equivalent (cps)")

        if np.isfinite(dwell_s):
            ax_top.set_title(
                f"Sampling geometry | QMS N={display_n}, dwell={dwell_s * 1e3:.3g} ms"
            )
        else:
            ax_top.set_title(f"Sampling geometry | QMS N={display_n}: invalid dwell")

        ax_top.grid(True, alpha=0.25)
        ax_top.legend(fontsize=8, loc="upper right", ncol=2)

        ax_bot.set_xlabel("Time (s)")
        ax_bot.set_ylabel("Counts per data point")
        ax_bot.grid(True, alpha=0.25)
        ax_bot.legend(fontsize=8, loc="upper right")
        ax_bot.set_xlim(zoom_start, zoom_end)

        self.export_data["2. Sampling Geometry"] = rows

        self.fig_sampling.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.10, hspace=0.30)
        self.canvas_sampling.draw()

    # --------------------------------------------------------
    # TAB 3
    # THREE-SWEEP, SMOOTH AVERAGE-DUTY-CYCLE QMS-ToF COMPARISON
    # --------------------------------------------------------

    def tab3_total_rsd(self, signal_counts, background_counts, dose_shots):
        """
        Modeled RSD for one representative element in one nominal image pixel.

        Counting variance uses signal + background, while the denominator is
        analyte signal only. Shot-yield and transport variability are treated
        as independent pulse-to-pulse terms and decrease as 1/sqrt(dose).
        """
        signal_counts = float(signal_counts)
        background_counts = max(0.0, float(background_counts))
        dose_shots = max(float(dose_shots), 1.0)

        if not np.isfinite(signal_counts) or signal_counts <= 0:
            return np.nan, np.nan, np.nan

        counting_rsd = 100.0 * np.sqrt(signal_counts + background_counts) / signal_counts

        pulse_rsd = np.sqrt(
            max(0.0, float(self.shot_yield_rsd_var.get())) ** 2
            + max(0.0, float(self.transport_rsd_var.get())) ** 2
        ) / np.sqrt(dose_shots)

        total_rsd = np.sqrt(counting_rsd ** 2 + pulse_rsd ** 2)
        return total_rsd, counting_rsd, pulse_rsd

    @staticmethod
    def tab3_qms_status(dwell_s, switch_s, min_dwell_s, sweep_s, pixel_time_s):
        """
        Return two independent labels for a QMS condition.

        dwell_status:
            favorable              dwell >= switching/settling time
            overhead_dominated     minimum dwell <= dwell < switching time
            below_minimum_dwell    dwell below the entered minimum
            invalid_or_zero_dwell  no positive dwell remains

        cycle_relation:
            fits_within_pixel      sweep < pixel residence time
            equals_pixel           sweep approximately equals pixel time
            longer_than_pixel      sweep exceeds pixel time; the plotted value
                                   is only a long-run average-duty-cycle estimate
                                   and excludes pixel-scale phase aliasing
        """
        if not np.isfinite(dwell_s) or dwell_s <= 0:
            dwell_status = "invalid_or_zero_dwell"
        elif dwell_s < min_dwell_s:
            dwell_status = "below_minimum_dwell"
        elif switch_s > 0 and dwell_s < switch_s:
            dwell_status = "overhead_dominated"
        else:
            dwell_status = "favorable"

        tol = 1e-12
        if sweep_s < pixel_time_s - tol:
            cycle_relation = "fits_within_pixel"
        elif abs(sweep_s - pixel_time_s) <= tol:
            cycle_relation = "equals_pixel"
        else:
            cycle_relation = "longer_than_pixel"

        return dwell_status, cycle_relation

    def shot_rsd_rows(self):
        """
        Tab 3 compares one representative element while the x-axis changes the
        total number of masses/elements included in the analytical method.

        ToF model
        ---------
        All masses inside the selected ToF window are measured quasi-
        simultaneously. The representative element therefore retains the same
        modeled counts as method size increases. The three horizontal lines
        differ only by the explicit restricted-mass-window count multiplier.

        QMS model
        ---------
        Separate 5, 10, and 20 ms sweep families are evaluated. For each sweep
        and N measured elements:

            dwell = (sweep - N * switch) / N
            live_fraction = dwell / sweep

        Each shaded band spans the entered relative QMS sensitivity bounds.
        Each central line uses the current Quad sensitivity multiplier. The
        calculation is a smooth long-run average-duty-cycle counting model.
        When sweep > pixel time, it does not model QMS/laser phase aliasing or
        guarantee that every nominal pixel contains a dwell for that element.
        """
        max_elements = max(1, int(self.max_elements_var.get()))
        dose = max(1, int(self.dose_var.get()))
        rep_hz = max(float(self.freq_var.get()), 1e-12)
        pixel_time_s = dose / rep_hz
        source_counts_per_shot = self.current_counts_per_shot(self.active_class_var.get())

        if not finite_positive(source_counts_per_shot):
            return []

        bkg_cps = max(0.0, float(self.background_cps_var.get()))
        switch_ms = max(float(self.quad_switch_ms_var.get()), 0.0)
        switch_s = switch_ms / 1000.0
        line_sensitivity = max(float(self.quad_sensitivity_mult_var.get()), 1e-12)
        min_dwell_s = max(float(self.tab3_min_dwell_ms_var.get()) / 1000.0, 0.0)

        sweep_values_ms = parse_positive_number_list(
            self.tab3_qms_sweeps_var.get(),
            "Tab 3 QMS sweeps",
        )
        sensitivity_values = parse_positive_number_list(
            self.tab3_qms_sensitivities_var.get(),
            "Tab 3 QMS sensitivity bounds",
        )
        sensitivity_low = min(sensitivity_values)
        sensitivity_high = max(sensitivity_values)

        rows = []

        # Three horizontal ToF mass-window references.
        for case in TOF_MASS_WINDOWS:
            gain = tof_mass_window_multiplier(case["low"], case["high"])
            signal_counts = dose * source_counts_per_shot * gain
            background_counts = bkg_cps * pixel_time_s * gain
            total_rsd, counting_rsd, pulse_rsd = self.tab3_total_rsd(
                signal_counts,
                background_counts,
                dose,
            )

            for n_elements in range(1, max_elements + 1):
                rows.append({
                    "instrument": "ToF",
                    "case": case["label"],
                    "n_elements": n_elements,
                    "pixel_time_ms": pixel_time_s * 1000.0,
                    "dose_shots_per_pixel": dose,
                    "source_counts_per_shot": source_counts_per_shot,
                    "signal_counts_per_pixel": signal_counts,
                    "background_counts_per_pixel": background_counts,
                    "element_pixel_rsd_percent": total_rsd,
                    "pixel_rsd_percent": total_rsd,
                    "counting_rsd_percent": counting_rsd,
                    "pulse_variability_rsd_percent": pulse_rsd,
                    "tof_multiplier": gain,
                    "tof_mass_low": case["low"],
                    "tof_mass_high": case["high"],
                    "qms_sweep_ms": np.nan,
                    "qms_switch_ms": np.nan,
                    "qms_sensitivity_multiplier": np.nan,
                    "qms_dwell_ms": np.nan,
                    "qms_live_fraction": np.nan,
                    "dwell_status": "tof_quasi_simultaneous_reference",
                    "cycle_relation": "tof_quasi_simultaneous_reference",
                    "is_qms_line": False,
                    "is_qms_bound": False,
                    "modeled": True,
                })

        # Three QMS sweep families. Only the two sensitivity bounds and the
        # selected line sensitivity are required to draw each field and line.
        qms_sensitivities = sorted(set([
            sensitivity_low,
            sensitivity_high,
            line_sensitivity,
        ]))

        for sweep_ms in sweep_values_ms:
            sweep_s = sweep_ms / 1000.0

            for sensitivity in qms_sensitivities:
                is_line = abs(sensitivity - line_sensitivity) < 1e-12
                is_bound = (
                    abs(sensitivity - sensitivity_low) < 1e-12
                    or abs(sensitivity - sensitivity_high) < 1e-12
                )

                for n_elements in range(1, max_elements + 1):
                    dwell_s = quad_dwell_time_per_element(
                        sweep_s,
                        switch_s,
                        n_elements,
                    )
                    dwell_status, cycle_relation = self.tab3_qms_status(
                        dwell_s,
                        switch_s,
                        min_dwell_s,
                        sweep_s,
                        pixel_time_s,
                    )
                    modeled = dwell_status in {"favorable", "overhead_dominated"}

                    if modeled:
                        live_fraction = dwell_s / sweep_s
                        signal_counts = (
                            dose
                            * source_counts_per_shot
                            * sensitivity
                            * live_fraction
                        )
                        background_counts = (
                            bkg_cps
                            * pixel_time_s
                            * sensitivity
                            * live_fraction
                        )
                        total_rsd, counting_rsd, pulse_rsd = self.tab3_total_rsd(
                            signal_counts,
                            background_counts,
                            dose,
                        )
                    else:
                        live_fraction = np.nan
                        signal_counts = np.nan
                        background_counts = np.nan
                        total_rsd = counting_rsd = pulse_rsd = np.nan

                    rows.append({
                        "instrument": "QMS",
                        "case": f"QMS {sweep_ms:g} ms sweep",
                        "n_elements": n_elements,
                        "pixel_time_ms": pixel_time_s * 1000.0,
                        "dose_shots_per_pixel": dose,
                        "source_counts_per_shot": source_counts_per_shot,
                        "signal_counts_per_pixel": signal_counts,
                        "background_counts_per_pixel": background_counts,
                        "element_pixel_rsd_percent": total_rsd,
                        "pixel_rsd_percent": total_rsd,
                        "counting_rsd_percent": counting_rsd,
                        "pulse_variability_rsd_percent": pulse_rsd,
                        "tof_multiplier": np.nan,
                        "tof_mass_low": np.nan,
                        "tof_mass_high": np.nan,
                        "qms_sweep_ms": sweep_ms,
                        "qms_switch_ms": switch_ms,
                        "qms_sensitivity_multiplier": sensitivity,
                        "qms_dwell_ms": dwell_s * 1000.0 if np.isfinite(dwell_s) else np.nan,
                        "qms_live_fraction": live_fraction,
                        "dwell_status": dwell_status,
                        "cycle_relation": cycle_relation,
                        "is_qms_line": is_line,
                        "is_qms_bound": is_bound,
                        "modeled": modeled,
                    })

        for row in rows:
            row["metric_name"] = (
                "Modeled RSD per nominal image pixel for one representative element"
            )
            row["metric_definition"] = (
                "Smooth average-duty-cycle counting model; x is the total number "
                "of masses/elements in the analytical method"
            )
            row["tab3_model_scope"] = (
                "Three QMS sweep families; fields vary relative QMS sensitivity; "
                "sweep-longer-than-pixel cases are long-run averages only and omit "
                "pixel-scale cycle-phase aliasing"
            )

        return rows

    @staticmethod
    def tab3_qms_envelope_for_sweep(rows, n_axis, sweep_ms):
        """Return low/high QMS RSD across the sensitivity-bound rows."""
        low = np.full(n_axis.size, np.nan, dtype=float)
        high = np.full(n_axis.size, np.nan, dtype=float)

        for j, n_elements in enumerate(n_axis.astype(int)):
            vals = [
                float(r["element_pixel_rsd_percent"])
                for r in rows
                if r["instrument"] == "QMS"
                and abs(float(r["qms_sweep_ms"]) - float(sweep_ms)) < 1e-12
                and r["n_elements"] == n_elements
                and r["is_qms_bound"]
                and r["modeled"]
                and np.isfinite(r["element_pixel_rsd_percent"])
            ]
            if vals:
                low[j] = min(vals)
                high[j] = max(vals)

        return low, high

    @staticmethod
    def tab3_qms_line_for_sweep(rows, n_axis, sweep_ms):
        """Return the selected-sensitivity QMS line and timing labels."""
        y = np.full(n_axis.size, np.nan, dtype=float)
        dwell_ms = np.full(n_axis.size, np.nan, dtype=float)
        dwell_status = np.full(n_axis.size, "", dtype=object)
        cycle_relation = np.full(n_axis.size, "", dtype=object)

        for j, n_elements in enumerate(n_axis.astype(int)):
            matches = [
                r for r in rows
                if r["instrument"] == "QMS"
                and abs(float(r["qms_sweep_ms"]) - float(sweep_ms)) < 1e-12
                and r["n_elements"] == n_elements
                and r["is_qms_line"]
            ]
            if not matches:
                continue

            row = matches[0]
            dwell_ms[j] = row["qms_dwell_ms"]
            dwell_status[j] = row["dwell_status"]
            cycle_relation[j] = row["cycle_relation"]

            if row["modeled"] and np.isfinite(row["element_pixel_rsd_percent"]):
                y[j] = float(row["element_pixel_rsd_percent"])

        return y, dwell_ms, dwell_status, cycle_relation

    def update_shot_rsd_tab(self):
        ax = self.ax_shot
        ax.clear()

        rows = self.shot_rsd_rows()

        if not rows:
            ax.text(
                0.5,
                0.5,
                "No element-specific image-pixel RSD comparison for the selected settings.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            self.canvas_shot.draw()
            return

        max_elements = max(r["n_elements"] for r in rows)
        n_axis = np.arange(1, max_elements + 1, dtype=float)

        sweep_values_ms = parse_positive_number_list(
            self.tab3_qms_sweeps_var.get(),
            "Tab 3 QMS sweeps",
        )
        sensitivity_values = parse_positive_number_list(
            self.tab3_qms_sensitivities_var.get(),
            "Tab 3 QMS sensitivity bounds",
        )
        sensitivity_low = min(sensitivity_values)
        sensitivity_high = max(sensitivity_values)

        switch_ms = float(self.quad_switch_ms_var.get())
        line_sensitivity = float(self.quad_sensitivity_mult_var.get())
        min_dwell_ms = float(self.tab3_min_dwell_ms_var.get())
        dose = max(1, int(self.dose_var.get()))
        rep_hz = max(float(self.freq_var.get()), 1e-12)
        pixel_time_ms = 1000.0 * dose / rep_hz

        # QMS fields and selected-sensitivity lines. Fields are drawn first so
        # the six reference lines remain crisp and editable in the PDF.
        qms_line_cache = {}
        for i, sweep_ms in enumerate(sweep_values_ms):
            style = TAB3_QMS_STYLE_CYCLE[i % len(TAB3_QMS_STYLE_CYCLE)]
            low, high = self.tab3_qms_envelope_for_sweep(rows, n_axis, sweep_ms)
            mask = np.isfinite(low) & np.isfinite(high)

            relation_note = (
                "long-run average only"
                if sweep_ms > pixel_time_ms + 1e-12
                else "fits/equal to pixel"
            )

            if np.any(mask):
                fill = ax.fill_between(
                    n_axis,
                    low,
                    high,
                    where=mask,
                    color=style["color"],
                    alpha=style["alpha"],
                    interpolate=False,
                    linewidth=0.8,
                    edgecolor=style["color"],
                    label=(
                        f"QMS {sweep_ms:g} ms field: {sensitivity_low:g}-"
                        f"{sensitivity_high:g}x sensitivity ({relation_note})"
                    ),
                    zorder=1,
                )
                if style.get("hatch"):
                    fill.set_hatch(style["hatch"])

            y, dwell_ms, dwell_status, cycle_relation = self.tab3_qms_line_for_sweep(
                rows,
                n_axis,
                sweep_ms,
            )
            qms_line_cache[sweep_ms] = (y, dwell_ms, dwell_status, cycle_relation)

            line_mask = np.isfinite(y)
            if np.any(line_mask):
                ax.plot(
                    n_axis[line_mask],
                    y[line_mask],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.8,
                    marker="o",
                    markersize=4.0,
                    markerfacecolor=style["color"],
                    markeredgecolor=style["color"],
                    label=(
                        f"QMS {sweep_ms:g} ms line at {line_sensitivity:g}x sensitivity"
                    ),
                    zorder=4,
                )

                overhead = line_mask & (dwell_status == "overhead_dominated")
                if np.any(overhead):
                    ax.plot(
                        n_axis[overhead],
                        y[overhead],
                        linestyle="none",
                        marker="o",
                        markersize=5.2,
                        markerfacecolor="white",
                        markeredgecolor=style["color"],
                        markeredgewidth=1.2,
                        zorder=5,
                    )

        # Three horizontal ToF mass-window references.
        for case in TOF_MASS_WINDOWS:
            sub = [
                r for r in rows
                if r["instrument"] == "ToF" and r["case"] == case["label"]
            ]
            sub.sort(key=lambda r: r["n_elements"])
            if len(sub) != max_elements:
                continue

            y = np.array([r["element_pixel_rsd_percent"] for r in sub], dtype=float)
            gain = tof_mass_window_multiplier(case["low"], case["high"])
            ax.plot(
                n_axis,
                y,
                color=case["color"],
                linestyle=case["linestyle"],
                linewidth=2.4,
                label=f"{case['label']} ({gain:.2f}x modeled counts)",
                zorder=3,
            )

        ax.set_xlim(1, max_elements)
        ax.set_xticks(np.arange(1, max_elements + 1, dtype=int))
        ax.set_xlabel("Total number of measured masses / elements in the method")
        ax.set_ylabel("Modeled RSD per nominal image pixel for one element (%)")
        ax.set_title(
            "One-element precision versus total method size: three QMS sweep conditions\n"
            f"QMS fields={sensitivity_low:g}-{sensitivity_high:g}x relative sensitivity; "
            f"QMS lines={line_sensitivity:g}x; switch={switch_ms:g} ms; "
            f"pixel time={pixel_time_ms:g} ms",
            fontsize=10.5,
        )
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.0, loc="upper left", frameon=True, ncol=2)

        # Fixed publication/export range requested by the user. Values above
        # 20% are intentionally clipped so repeated exports use the same scale.
        ax.set_ylim(0.0, 20.0)

        long_sweeps = [s for s in sweep_values_ms if s > pixel_time_ms + 1e-12]
        long_note = (
            ", ".join(f"{s:g} ms" for s in long_sweeps)
            if long_sweeps
            else "none"
        )

        ax.text(
            0.99,
            0.015,
            (
                "Model assumptions: QMS counts use average live fraction = dwell/sweep. "
                "Fields vary only the relative sensitivity multiplier.\n"
                f"Filled markers: dwell >= switch; open markers: {min_dwell_ms:g} ms <= "
                f"dwell < {switch_ms:g} ms; each curve stops below {min_dwell_ms:g} ms. "
                f"Sweep > pixel ({long_note}) is a long-run average only and excludes "
                "pixel-scale phase/aliasing effects."
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.2,
            color="0.25",
        )

        self.export_data["3. Element-Specific Pixel RSD"] = rows

        self.fig_shot.subplots_adjust(
            left=0.09,
            right=0.98,
            top=0.86,
            bottom=0.15,
        )
        self.canvas_shot.draw()

    # --------------------------------------------------------
    # TAB 4
    # --------------------------------------------------------

    def pixel_rsd_rows(self):
        """
        Pixel-pixel RSD vs source counts per laser pulse.

        This calculates analyte precision.

        Denominator = signal counts per pixel only.
        Variance    = signal + background counting variance, plus deterministic signal aliasing.
        """
        _, pixel_starts_s, pixel_width_s, _, _ = self.pixel_window()

        if pixel_starts_s is None:
            return []

        current_counts = self.current_counts_per_shot(self.active_class_var.get())

        if not finite_positive(current_counts):
            return []

        counts_per_pulse_grid = np.logspace(-3, 5, 90)

        rows = []

        restricted_mult = max(0.0, float(self.tof_restricted_mult_var.get()))

        tof_cases = [
            ("ToF full range", 1.0),
            (f"ToF restricted range, {restricted_mult:g}x counts", restricted_mult),
        ]

        duration_s = float(self.duration_s_var.get())
        sweep_s = float(self.quad_sweep_ms_var.get()) / 1000.0
        switch_s = float(self.quad_switch_ms_var.get()) / 1000.0
        qmult = float(self.quad_sensitivity_mult_var.get())
        bkg_cps = float(self.background_cps_var.get())

        for counts_per_pulse in counts_per_pulse_grid:
            scaled_rate = self.scaled_rate_for_counts_per_pulse(counts_per_pulse)

            if scaled_rate is None:
                continue

            for tint_ms in TOF_TINTS_MS:
                tint_s = tint_ms / 1000.0

                for case_name, mult in tof_cases:

                    starts_sig, mu_signal, width = tof_events(
                        rate_cps=scaled_rate,
                        dt_s=self.dt_s,
                        duration_s=duration_s,
                        tint_s=tint_s,
                        background_cps=0.0,
                        signal_multiplier=mult,
                        background_multiplier=0.0,
                    )

                    starts_tot, mu_total, width_tot = tof_events(
                        rate_cps=scaled_rate,
                        dt_s=self.dt_s,
                        duration_s=duration_s,
                        tint_s=tint_s,
                        background_cps=bkg_cps,
                        signal_multiplier=mult,
                        background_multiplier=1.0,
                    )

                    signal_means, _ = event_counts_to_window_mean_var(
                        starts_sig,
                        width,
                        mu_signal,
                        pixel_starts_s,
                        pixel_width_s,
                    )

                    _, total_vars = event_counts_to_window_mean_var(
                        starts_tot,
                        width_tot,
                        mu_total,
                        pixel_starts_s,
                        pixel_width_s,
                    )

                    mean_ct, rsd, det_var, pois_var = analyte_pixel_rsd(
                        signal_means,
                        total_vars,
                    )

                    rows.append({
                        "instrument": "ToF",
                        "case": case_name,
                        "n_elements": 0,
                        "counts_per_laser_pulse_source": counts_per_pulse,
                        "signal_counts_per_pixel_mean": mean_ct,
                        "pixel_rsd_percent": rsd,
                        "deterministic_variance_signal": det_var,
                        "poisson_variance_signal_plus_background": pois_var,
                        "tint_ms": tint_ms,
                        "dwell_ms": np.nan,
                        "dose_shots_per_pixel": int(self.dose_var.get()),
                    })

            for n_elements in QMS_COMPARE_N:

                starts_sig, mu_signal, dwell_s = qms_events(
                    rate_cps_tof_sensitivity=scaled_rate,
                    dt_s=self.dt_s,
                    duration_s=duration_s,
                    n_elements=n_elements,
                    sweep_s=sweep_s,
                    switch_s=switch_s,
                    qms_sensitivity_mult=qmult,
                    element_index=0,
                    background_cps=0.0,
                )

                starts_tot, mu_total, dwell_s_tot = qms_events(
                    rate_cps_tof_sensitivity=scaled_rate,
                    dt_s=self.dt_s,
                    duration_s=duration_s,
                    n_elements=n_elements,
                    sweep_s=sweep_s,
                    switch_s=switch_s,
                    qms_sensitivity_mult=qmult,
                    element_index=0,
                    background_cps=bkg_cps,
                )

                if np.isfinite(dwell_s) and np.isfinite(dwell_s_tot):
                    signal_means, _ = event_counts_to_window_mean_var(
                        starts_sig,
                        dwell_s,
                        mu_signal,
                        pixel_starts_s,
                        pixel_width_s,
                    )

                    _, total_vars = event_counts_to_window_mean_var(
                        starts_tot,
                        dwell_s_tot,
                        mu_total,
                        pixel_starts_s,
                        pixel_width_s,
                    )

                    mean_ct, rsd, det_var, pois_var = analyte_pixel_rsd(
                        signal_means,
                        total_vars,
                    )
                else:
                    mean_ct, rsd, det_var, pois_var = np.nan, np.nan, np.nan, np.nan

                rows.append({
                    "instrument": "QMS",
                    "case": "QMS",
                    "n_elements": n_elements,
                    "counts_per_laser_pulse_source": counts_per_pulse,
                    "signal_counts_per_pixel_mean": mean_ct,
                    "pixel_rsd_percent": rsd,
                    "deterministic_variance_signal": det_var,
                    "poisson_variance_signal_plus_background": pois_var,
                    "tint_ms": np.nan,
                    "dwell_ms": dwell_s * 1000.0 if np.isfinite(dwell_s) else np.nan,
                    "dose_shots_per_pixel": int(self.dose_var.get()),
                    "quad_sensitivity_multiplier": qmult,
                })

        return rows

    def update_pixel_rsd_tab(self):
        ax = self.ax_pixel
        ax.clear()

        rows = self.pixel_rsd_rows()

        if not rows:
            ax.text(
                0.5,
                0.5,
                "Not enough pixels or valid signal.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            self.canvas_pixel.draw()
            return

        y_plot_max = 220.0

        q_colors = {
            1: "black",
            2: "C0",
            5: "C2",
            10: "C3",
            20: "C4",
        }

        for n_elements in QMS_COMPARE_N:
            sub = [
                r for r in rows
                if r["instrument"] == "QMS" and r["n_elements"] == n_elements
            ]

            x = np.array([r["counts_per_laser_pulse_source"] for r in sub], dtype=float)
            y = np.array([r["pixel_rsd_percent"] for r in sub], dtype=float)
            dwell = np.array([r["dwell_ms"] for r in sub], dtype=float)

            good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y <= y_plot_max)

            if not np.any(good):
                ax.plot(
                    [],
                    [],
                    color=q_colors.get(n_elements, None),
                    linewidth=2.0,
                    label=f"QMS N={n_elements}: off scale/invalid",
                )
                continue

            dwell_med = np.nanmedian(dwell)

            ax.plot(
                x[good],
                y[good],
                color=q_colors.get(n_elements, None),
                linewidth=2.3,
                label=f"QMS N={n_elements}, dwell={dwell_med:.3g} ms",
            )

        tint_colors = {
            0.1: "C5",
            1.0: "C6",
            5.0: "C7",
        }

        restricted_label = f"ToF restricted range, {self.tof_restricted_mult_var.get():g}x counts"

        for tint_ms in TOF_TINTS_MS:
            for case_name, linestyle in [
                ("ToF full range", "--"),
                (restricted_label, ":"),
            ]:
                sub = [
                    r for r in rows
                    if r["instrument"] == "ToF"
                    and abs(r["tint_ms"] - tint_ms) < 1e-12
                    and r["case"] == case_name
                ]

                x = np.array([r["counts_per_laser_pulse_source"] for r in sub], dtype=float)
                y = np.array([r["pixel_rsd_percent"] for r in sub], dtype=float)

                good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y <= y_plot_max)

                if np.any(good):
                    ax.plot(
                        x[good],
                        y[good],
                        color=tint_colors.get(tint_ms, None),
                        linestyle=linestyle,
                        linewidth=1.8,
                        label=f"{case_name}, Tint={tint_ms:g} ms",
                    )

        for threshold in [5, 10, 20, 50, 100, 200]:
            ax.axhline(
                threshold,
                color="gray",
                linestyle=":",
                linewidth=0.8,
                alpha=0.55,
            )

            ax.text(
                1.15e-3,
                threshold + 1,
                f"{threshold}%",
                fontsize=8,
                color="gray",
            )

        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1e5)
        ax.set_ylim(0, y_plot_max)

        ax.set_xlabel("Source counts per laser pulse for that element")
        ax.set_ylabel("Pixel-pixel analyte RSD (%)")

        ax.set_title(
            "Pixel-pixel analyte precision vs source counts per pulse\n"
            "Denominator is signal only; variance includes signal + background"
        )

        ax.grid(True, which="major", alpha=0.30)
        ax.grid(True, which="minor", alpha=0.12)
        ax.legend(fontsize=7, loc="upper right", ncol=2)

        self.export_data["4. Pixel-Pixel RSD"] = rows

        self.fig_pixel.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.12)
        self.canvas_pixel.draw()

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    def export_current_tab_csv(self):
        tab_text = self.notebook.tab(self.notebook.select(), "text")
        rows = self.export_data.get(tab_text, [])

        if not rows:
            messagebox.showinfo(
                "Export CSV",
                f"No exportable data found for:\n{tab_text}\n\nClick UPDATE first.",
            )
            return

        clean_name = (
            tab_text.lower()
            .replace(" ", "_")
            .replace(".", "")
            .replace("+", "plus")
            .replace("#", "num")
            .replace("/", "_")
            .replace("%", "pct")
        )

        filename = filedialog.asksaveasfilename(
            title="Save current tab data as CSV",
            defaultextension=".csv",
            initialfile=clean_name + ".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )

        if not filename:
            return

        try:
            fieldnames = sorted(set().union(*(row.keys() for row in rows)))

            with open(filename, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            messagebox.showinfo(
                "Export CSV",
                f"Saved {len(rows)} rows to:\n{filename}",
            )

        except Exception as e:
            messagebox.showerror("Export CSV error", str(e))


    def export_current_tab_pdf(self):
        """Export the selected plot as an editable vector PDF for Adobe apps."""
        tab_text = self.notebook.tab(self.notebook.select(), "text")
        figure_map = {
            "1. Raw LA Signal": self.fig_signal,
            "2. Sampling Geometry": self.fig_sampling,
            "3. Element-Specific Pixel RSD": self.fig_shot,
            "4. Pixel-Pixel RSD": self.fig_pixel,
        }
        fig = figure_map.get(tab_text)

        if fig is None:
            messagebox.showinfo("Export PDF", f"No figure found for:\n{tab_text}")
            return

        clean_name = (
            tab_text.lower()
            .replace(" ", "_")
            .replace(".", "")
            .replace("+", "plus")
            .replace("#", "num")
            .replace("/", "_")
            .replace("%", "pct")
        )

        filename = filedialog.asksaveasfilename(
            title="Save current tab as editable vector PDF",
            defaultextension=".pdf",
            initialfile=clean_name + ".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )

        if not filename:
            return

        try:
            fig.savefig(
                filename,
                format="pdf",
                bbox_inches="tight",
                metadata={
                    "Title": tab_text,
                    "Subject": "Vector figure exported from Quad vs ToF model",
                },
            )
            messagebox.showinfo(
                "Export PDF",
                "Saved an editable vector PDF to:\n" + filename,
            )
        except Exception as e:
            messagebox.showerror("Export PDF error", str(e))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = QuadVsToFApp(root)
    root.mainloop()
