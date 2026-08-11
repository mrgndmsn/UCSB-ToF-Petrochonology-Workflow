import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.signal import convolve
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

plt.style.use("fast")

# ============================================================
# CONSTANTS
# ============================================================
TINT_MIN_MS = 0.01
TINT_MAX_MS = 5.0
FAST_TINT_N = 160

DT_US = 10.0
INTERNAL_GATE_HYSTERESIS = 0.50

OVERLAY_DOSES = [1, 10, 20]
OVERLAY_WASHOUTS_MS = [1.0, 2.0, 5.0]
REP_RATES = [250.0, 500.0, 750.0, 1000.0]
SNAPSHOT_TINTS_MS = [0.1, 0.5, 0.82, 1.0, 1.32, 2.0]

MASTER_RSD_YMAX = 100.0
SINGLE_PIXEL_RSD_YMAX = 100.0

CONC_CLASSES = ["Ultra trace", "Trace", "Minor", "Major"]

SHAPE_PRESETS = {
    "compact": {"rise_frac": 0.06, "slow_frac": 0.05, "slow_tau_mult": 1.4},
    "broad": {"rise_frac": 0.45, "slow_frac": 0.10, "slow_tau_mult": 1.6},
    "tail-heavy": {"rise_frac": 0.05, "slow_frac": 0.50, "slow_tau_mult": 6.0},
}
SHAPE_ORDER = ["compact", "broad", "tail-heavy"]

DOSE_COLORS = {1: "C0", 2: "C4", 5: "C1", 10: "C2", 20: "C3"}
WASH_LINESTYLES = {1.0: "-", 2.0: "--", 5.0: ":"}
WASH_MARKERS = {1.0: "o", 2.0: "s", 5.0: "^"}
REP_SHADE = {250.0: 0.25, 500.0: 0.55, 750.0: 0.75, 1000.0: 1.00}

CLASS_COLORS = {
    "Ultra trace": "C0",
    "Trace": "C1",
    "Minor": "C2",
    "Major": "C3",
}
EDGE_FOR_CLASS = {
    "Ultra trace": "blue",
    "Trace": "green",
    "Minor": "gold",
    "Major": "red",
}
SHAPE_COLORS = {
    "compact": "C0",
    "broad": "C2",
    "tail-heavy": "C3",
}
SHAPE_LINESTYLES = {
    "compact": "-",
    "broad": "--",
    "tail-heavy": ":",
}

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


def mean_by_mode(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.mean(x))


def shade_color(color, factor):
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    white = np.ones(3, dtype=float)
    return tuple(np.clip(white * (1.0 - factor) + rgb * factor, 0, 1))


def clamp_pixel_window(start_pix_req, end_pix_req, n_pix_total, min_pixels=3):
    if n_pix_total < min_pixels:
        return None, None

    start = max(0, int(start_pix_req))
    end = min(n_pix_total, max(start + 1, int(end_pix_req)))

    if end - start < min_pixels:
        start = max(0, end - min_pixels)
        end = min(n_pix_total, start + min_pixels)

    if end - start < min_pixels:
        return None, None

    return start, end


def legend_right_at(ax, y=0.5, **kwargs):
    return ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, y),
        borderaxespad=0.0,
        frameon=True,
        **kwargs
    )


def tight_layout_for_right_legends(fig, right=0.78, pad=2.0):
    try:
        fig.tight_layout(rect=[0.0, 0.0, right, 1.0], pad=pad)
    except Exception:
        pass


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
# APPLICATION
# ============================================================
class LaTofSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("LA-TOF-MS Simulator")
        self.root.geometry("1540x1040")
        self._set_fonts()

        self.export_frames = {}

        # primary controls
        self.freq_var = tk.DoubleVar(value=500.0)
        self.width_var = tk.DoubleVar(value=2.0)
        self.int_time_var = tk.DoubleVar(value=0.1)
        self.report_tint_ms_var = tk.DoubleVar(value=0.1)
        self.duration_var = tk.DoubleVar(value=0.5)

        self.spot_um_var = tk.DoubleVar(value=5.0)
        self.depth_per_shot_um_var = tk.DoubleVar(value=0.1)

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

        self.start_pix_var = tk.IntVar(value=10)
        self.end_pix_var = tk.IntVar(value=200)
        self.dose_var = tk.IntVar(value=5)

        self.shot_yield_rsd_var = tk.DoubleVar(value=5.0)
        self.transport_rsd_var = tk.DoubleVar(value=0.0)
        self.shot_jitter_us_var = tk.DoubleVar(value=0.0)

        self.bkg_rate_cps_var = tk.DoubleVar(value=10.0)
        self.bkg_sigma_mult_var = tk.DoubleVar(value=3.0)
        self.bkg_gate_mode_var = tk.IntVar(value=0)
        self.bkg_min_run_bins_var = tk.IntVar(value=2)
        self.bkg_drift_rsd_var = tk.DoubleVar(value=0.0)
        self.bkg_corr_ms_var = tk.DoubleVar(value=2.0)

        self.show_theory_var = tk.IntVar(value=1)
        self.show_lod_var = tk.IntVar(value=1)
        self.lod_sigma_var = tk.DoubleVar(value=3.0)

        self.overlay_washouts_ms = list(OVERLAY_WASHOUTS_MS)
        self.overlay_doses = list(OVERLAY_DOSES)
        self.rep_rates = list(REP_RATES)
        self.snapshot_tints_ms = list(SNAPSHOT_TINTS_MS)

        self.snapshot_start_shot = 10
        self.snapshot_nshots = 20

        self.stat_mean = tk.StringVar(value="--")
        self.stat_max = tk.StringVar(value="--")
        self.stat_rsd = tk.StringVar(value="--")
        self.stat_total = tk.StringVar(value="--")
        self.stat_shot_mean = tk.StringVar(value="--")
        self.stat_shot_rsd = tk.StringVar(value="--")
        self.stat_pix_mean = tk.StringVar(value="--")
        self.stat_pix_rsd = tk.StringVar(value="--")

        self._build_ui()
        self.update_full()

    def _set_fonts(self):
        self.root.option_add("*Font", "Arial 8")
        self.root.option_add("*TCombobox*Listbox*Font", "Arial 8")

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------
    def _build_ui(self):
        top_container = tk.Frame(self.root, bg="#f0f0f0", pady=3)
        top_container.pack(side=tk.TOP, fill=tk.X)

        control_frame = ttk.LabelFrame(top_container, text="Parameters")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        r = 0
        c = 0
        self.create_slider(control_frame, "Laser Freq (Hz)", self.freq_var, 1, 1000, r, c); c += 1
        self.create_slider(control_frame, "Washout (ms)", self.width_var, 0.5, 10.0, r, c); c += 1
        self.create_slider(control_frame, "Integration (ms)", self.int_time_var, 0.01, 100.0, r, c); c += 1
        self.create_slider(control_frame, "Report Tint (ms)", self.report_tint_ms_var, 0.01, 10.0, r, c); c += 1
        self.create_slider(control_frame, "Spot (µm)", self.spot_um_var, 1.0, 50.0, r, c); c += 1
        self.create_slider(control_frame, "Depth/shot (µm)", self.depth_per_shot_um_var, 0.01, 1.0, r, c); c += 1
        self.create_slider(control_frame, "Acq Time (s)", self.duration_var, 0.1, 2.0, r, c); c += 1
        self.create_entry_int(control_frame, "Start pix", self.start_pix_var, r, c); c += 1
        self.create_entry_int(control_frame, "End pix", self.end_pix_var, r, c); c += 1
        self.create_slider(control_frame, "Dose (shots/pix)", self.dose_var, 1, 50, r, c); c += 1

        class_frame = ttk.Frame(control_frame)
        class_frame.grid(row=r, column=c, padx=6, pady=2, sticky="ew")
        ttk.Label(class_frame, text="Signal class:", font=("Arial", 8)).pack(side=tk.LEFT)
        ttk.Combobox(
            class_frame,
            textvariable=self.active_class_var,
            values=CONC_CLASSES,
            state="readonly",
            width=12
        ).pack(side=tk.LEFT, padx=(6, 0))

        r = 1
        c2 = 0
        self.create_slider(control_frame, "Shot yield RSD (%)", self.shot_yield_rsd_var, 0.0, 50.0, r, c2); c2 += 1
        self.create_slider(control_frame, "Transport RSD (%)", self.transport_rsd_var, 0.0, 50.0, r, c2); c2 += 1
        self.create_slider(control_frame, "Shot jitter (µs)", self.shot_jitter_us_var, 0.0, 200.0, r, c2); c2 += 1

        shape_frame = ttk.Frame(control_frame)
        shape_frame.grid(row=r, column=c2, padx=6, pady=2, sticky="ew")
        ttk.Label(shape_frame, text="Shape:", font=("Arial", 8)).pack(side=tk.LEFT)
        for lbl in SHAPE_ORDER:
            ttk.Radiobutton(shape_frame, text=lbl, variable=self.shape_profile_var, value=lbl).pack(side=tk.LEFT, padx=(4, 0))
        c2 += 1

        mode_frame = ttk.Frame(control_frame)
        mode_frame.grid(row=r, column=c2, padx=6, pady=2, sticky="ew")
        ttk.Label(mode_frame, text="Signal mode:", font=("Arial", 8)).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Class-based", variable=self.signal_mode_var, value="Class-based").pack(side=tk.LEFT, padx=(4, 0))
        ttk.Radiobutton(mode_frame, text="Manual cts/shot", variable=self.signal_mode_var, value="Manual cts/shot").pack(side=tk.LEFT, padx=(4, 0))
        ttk.Entry(mode_frame, textvariable=self.manual_counts_per_shot_var, width=8).pack(side=tk.LEFT, padx=(6, 0))

        r = 2
        c3 = 0
        self.create_slider(control_frame, "Background (cps)", self.bkg_rate_cps_var, 0, 5000, r, c3); c3 += 1
        self.create_slider(control_frame, "Gate σ-mult", self.bkg_sigma_mult_var, 0, 10, r, c3); c3 += 1
        self.create_entry_int(control_frame, "Min run bins", self.bkg_min_run_bins_var, r, c3); c3 += 1
        self.create_slider(control_frame, "Bkg drift RSD (%)", self.bkg_drift_rsd_var, 0.0, 100.0, r, c3); c3 += 1
        self.create_slider(control_frame, "Bkg corr time (ms)", self.bkg_corr_ms_var, 0.1, 20.0, r, c3); c3 += 1

        gate_frame = ttk.Frame(control_frame)
        gate_frame.grid(row=r, column=c3, padx=6, pady=2, sticky="ew")
        ttk.Label(gate_frame, text="Gate:", font=("Arial", 8)).pack(side=tk.LEFT)
        ttk.Radiobutton(gate_frame, text="Off", variable=self.bkg_gate_mode_var, value=0).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Radiobutton(gate_frame, text="Gate", variable=self.bkg_gate_mode_var, value=1).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Radiobutton(gate_frame, text="Gate+Sub", variable=self.bkg_gate_mode_var, value=2).pack(side=tk.LEFT, padx=(4, 0))
        c3 += 1

        theory_frame = ttk.Frame(control_frame)
        theory_frame.grid(row=r, column=c3, padx=6, pady=2, sticky="ew")
        ttk.Checkbutton(theory_frame, text="Show theory", variable=self.show_theory_var).pack(side=tk.LEFT)
        ttk.Checkbutton(theory_frame, text="Show LOD/SNR", variable=self.show_lod_var).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(theory_frame, text="LOD σ:", font=("Arial", 8)).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Entry(theory_frame, textvariable=self.lod_sigma_var, width=5).pack(side=tk.LEFT)

        r = 3
        c4 = 0
        self.create_entry(control_frame, "Ref conc (ppm)", self.ref_conc_ppm_var, r, c4); c4 += 1
        self.create_entry(control_frame, "Ref signal (cps)", self.ref_cps_var, r, c4); c4 += 1
        self.create_entry(control_frame, "Ref rep rate (Hz)", self.ref_rep_hz_var, r, c4); c4 += 1
        self.create_entry(control_frame, "Ref spot (µm)", self.ref_spot_um_var, r, c4); c4 += 1
        self.create_entry(control_frame, "Ref depth (µm)", self.ref_depth_um_var, r, c4); c4 += 1

        r = 4
        c5 = 0
        self.create_entry(control_frame, "Ultra trace (ppm)", self.ultra_trace_ppm_var, r, c5); c5 += 1
        self.create_entry(control_frame, "Trace (ppm)", self.trace_ppm_var, r, c5); c5 += 1
        self.create_entry(control_frame, "Minor (ppm)", self.minor_ppm_var, r, c5); c5 += 1
        self.create_entry(control_frame, "Major (ppm)", self.major_ppm_var, r, c5); c5 += 1

        for i in range(14):
            control_frame.columnconfigure(i, weight=1)

        btn_frame = tk.Frame(top_container, bg="#f0f0f0", pady=2)
        btn_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_fast = tk.Button(
            btn_frame, text="UPDATE (Main Tabs)",
            command=self.update_fast,
            bg="#e6f2ff", font=("Arial", 10, "bold"), height=1, width=22
        )
        self.btn_fast.pack(side=tk.LEFT, padx=(12, 6), pady=2)

        self.btn_full = tk.Button(
            btn_frame, text="UPDATE (ALL)",
            command=self.update_full,
            bg="#e1e1e1", font=("Arial", 10, "bold"), height=1, width=18
        )
        self.btn_full.pack(side=tk.LEFT, padx=(6, 12), pady=2)

        self.btn_export = tk.Button(
            btn_frame, text="Export Current Tab CSV",
            command=self.export_current_tab_csv,
            bg="#f5f5f5", font=("Arial", 9, "bold"), height=1, width=24
        )
        self.btn_export.pack(side=tk.LEFT, padx=(6, 12), pady=2)

        stats_frame = tk.Frame(self.root, bg="#e1e4e8", height=65)
        stats_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        def make_stat(parent, label, var, color="#007acc"):
            f = tk.Frame(parent, bg="#e1e4e8", bd=1, relief=tk.RIDGE)
            f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
            tk.Label(f, text=label, bg="#e1e4e8", fg="#555", font=("Arial", 8, "bold")).pack(pady=2)
            tk.Label(f, textvariable=var, bg="#e1e4e8", fg=color, font=("Arial", 12, "bold")).pack(pady=2)

        make_stat(stats_frame, "Mean Counts/Bin", self.stat_mean)
        make_stat(stats_frame, "Max Counts/Bin", self.stat_max, color="#D55E00")
        make_stat(stats_frame, "% RSD", self.stat_rsd)
        make_stat(stats_frame, "Total Sum", self.stat_total)
        make_stat(stats_frame, "Shot Mean", self.stat_shot_mean, color="#0072B2")
        make_stat(stats_frame, "Shot RSD (%)", self.stat_shot_rsd, color="#0072B2")
        make_stat(stats_frame, "Pixel Mean", self.stat_pix_mean, color="#009E73")
        make_stat(stats_frame, "Pixel RSD (%)", self.stat_pix_rsd, color="#009E73")

        self.plot_container = tk.Frame(self.root, bg="white", bd=2, relief=tk.SUNKEN)
        self.plot_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.notebook = ttk.Notebook(self.plot_container)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.tab_signal = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_signal, text="Signal")
        self.fig_sig, self.ax_sig, self.canvas_sig = self._make_plot_tab(self.tab_signal)

        self.tab_rsd = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_rsd, text="Pixel RSD vs Integration")
        self.fig_rsd, self.ax_rsd, self.canvas_rsd = self._make_plot_tab(self.tab_rsd)

        self.tab_shape = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_shape, text="Shape Comparison")
        self.fig_shape, self.ax_shape, self.canvas_shape = self._make_plot_tab(self.tab_shape)

        self.tab_snap = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_snap, text="Integration Snapshot")
        self.fig_snap, self.ax_snap, self.canvas_snap = self._make_plot_tab(self.tab_snap)

        self.tab_snap_thresh = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_snap_thresh, text="Integration Snapshot + Threshold")
        self.fig_snap_thresh, self.ax_snap_thresh, self.canvas_snap_thresh = self._make_plot_tab(self.tab_snap_thresh)

        self.tab_master = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_master, text="Pixel-to-Pixel Precision")
        self.fig_master, self.ax_master, self.canvas_master = self._make_plot_tab(self.tab_master)

        self.tab_singlepix = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_singlepix, text="Single-Pixel Counting Precision")
        self.fig_singlepix, self.ax_singlepix, self.canvas_singlepix = self._make_plot_tab(self.tab_singlepix)

    def _make_plot_tab(self, parent):
        frame = tk.Frame(parent, bg="white")
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(frame, bg="#ccc")
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        fig, ax = plt.subplots(figsize=(11.5, 6.5))
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        return fig, ax, canvas

    def create_slider(self, parent, label, variable, min_val, max_val, row, col):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, padx=6, pady=2, sticky="ew")
        ttk.Label(frame, text=label, font=("Arial", 8)).grid(row=0, column=0, sticky="w")
        scale = ttk.Scale(frame, from_=min_val, to=max_val, variable=variable, orient=tk.HORIZONTAL)
        scale.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        entry = ttk.Entry(frame, textvariable=variable, width=8)
        entry.grid(row=0, column=2, sticky="e")
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

    def export_current_tab_csv(self):
        tab_text = self.notebook.tab(self.notebook.select(), "text")
        df = self.export_frames.get(tab_text)
        if df is None or len(df) == 0:
            messagebox.showinfo("Export CSV", f"No exportable data available for '{tab_text}'.")
            return

        filename = filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            initialfile=tab_text.lower().replace(" ", "_") + ".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not filename:
            return

        try:
            df.to_csv(filename, index=False)
            messagebox.showinfo("Export CSV", f"Saved {len(df)} rows to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Export CSV", str(e))

    # --------------------------------------------------------
    # signal helpers
    # --------------------------------------------------------
    def concentration_map(self):
        return class_concentration_map(
            self.ultra_trace_ppm_var.get(),
            self.trace_ppm_var.get(),
            self.minor_ppm_var.get(),
            self.major_ppm_var.get(),
        )

    def current_class_conc_ppm(self, class_name=None):
        if class_name is None:
            class_name = self.active_class_var.get()
        return self.concentration_map()[class_name]

    def class_counts_per_shot(self, class_name=None):
        if class_name is None:
            class_name = self.active_class_var.get()

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

    def current_counts_per_shot(self, class_name=None):
        if str(self.signal_mode_var.get()) == "Manual cts/shot":
            return max(0.0, float(self.manual_counts_per_shot_var.get()))
        return self.class_counts_per_shot(class_name=class_name)

    def current_signal_cps(self, class_name=None, rep_hz=None):
        if class_name is None:
            class_name = self.active_class_var.get()
        rep_hz = float(self.freq_var.get()) if rep_hz is None else float(rep_hz)
        return self.current_counts_per_shot(class_name=class_name) * max(rep_hz, 0.0)

    def counts_per_um3_at_100(self, class_name=None):
        if class_name is None:
            class_name = self.active_class_var.get()

        if str(self.signal_mode_var.get()) == "Manual cts/shot":
            vol = shot_volume_um3(self.spot_um_var.get(), self.depth_per_shot_um_var.get())
            return max(0.0, float(self.manual_counts_per_shot_var.get())) / max(vol, 1e-30)

        return counts_per_um3_at_100_from_anchor_and_class(
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
        )

    # --------------------------------------------------------
    # background helpers
    # --------------------------------------------------------
    def _bkg_cps(self):
        return max(0.0, float(self.bkg_rate_cps_var.get()))

    def _bkg_counts_for_tint(self, Tint_s):
        return self._bkg_cps() * max(Tint_s, 0.0)

    def _report_tint_s(self):
        return max(float(self.report_tint_ms_var.get()) / 1000.0, DT_US / 1e6)

    def _shape_name(self, override=None):
        if override is not None:
            return str(override).strip().lower()
        return str(self.shape_profile_var.get()).strip().lower()

    def _scenario_seed(self, base, *vals):
        out = int(base)
        for i, v in enumerate(vals):
            out = (out * 1664525 + 1013904223 + int(round(float(v) * (11 + 2 * i)))) % (2**32 - 1)
        return int(out)

    # --------------------------------------------------------
    # core model
    # --------------------------------------------------------
    def _ideal_signal_highres(
        self,
        washout_override_ms=None,
        freq_override_hz=None,
        class_override=None,
        rng=None,
        shape_override=None,
    ):
        if rng is None:
            rng = np.random.default_rng(1)

        freq = float(self.freq_var.get()) if freq_override_hz is None else float(freq_override_hz)
        wash_ms = float(self.width_var.get()) if washout_override_ms is None else float(washout_override_ms)
        class_name = self.active_class_var.get() if class_override is None else str(class_override)
        duration = float(self.duration_var.get())

        counts_per_shot_mean = self.current_counts_per_shot(class_name=class_name)

        dt_sec = DT_US / 1e6
        time_high_res = np.arange(0.0, duration, dt_sec)
        n_hi = time_high_res.size
        if n_hi == 0:
            return np.array([]), np.array([]), dt_sec, "Count rate (cps)"

        Tshot = 1.0 / max(freq, 1e-12)
        shot_times = np.arange(0.0, duration, Tshot)
        n_shots = shot_times.size

        shot_factors = (
            sample_positive_factors(self.shot_yield_rsd_var.get(), n_shots, rng)
            * sample_positive_factors(self.transport_rsd_var.get(), n_shots, rng)
        )

        jitter_s = max(0.0, float(self.shot_jitter_us_var.get())) * 1e-6
        if jitter_s > 0:
            shot_times = shot_times + rng.normal(0.0, jitter_s, size=n_shots)

        shot_times = np.clip(shot_times, 0.0, max(duration - dt_sec, 0.0))
        shot_idx = np.clip((shot_times / dt_sec).astype(int), 0, n_hi - 1)

        triggers = np.zeros(n_hi, dtype=float)
        np.add.at(triggers, shot_idx, counts_per_shot_mean * shot_factors)

        _, kernel = washout_kernel_same_model(
            washout_ms=wash_ms,
            dt_sec=dt_sec,
            shape_name=self.shape_profile_var.get() if shape_override is None else shape_override,
        )

        karea = float(np.sum(kernel) * dt_sec)
        if karea <= 0:
            karea = 1.0

        ideal_rate_cps = convolve(triggers, kernel, mode="full")[:n_hi] / karea
        return time_high_res, np.maximum(ideal_rate_cps, 0.0), dt_sec, "Count rate (cps)"

    def _bin_expected_counts(self, ideal_rate_cps, dt_sec, duration, Tint_s):
        if Tint_s <= 0 or ideal_rate_cps.size == 0:
            return np.array([]), np.array([])

        bins = np.arange(0.0, duration + Tint_s, Tint_s)
        if bins[-1] < duration:
            bins = np.r_[bins, duration]

        cumsum = np.r_[0.0, np.cumsum(ideal_rate_cps) * dt_sec]
        sample_edges = np.arange(0.0, ideal_rate_cps.size + 1, 1.0) * dt_sec

        bin_idx = np.searchsorted(sample_edges, bins, side="left")
        bin_idx = np.clip(bin_idx, 0, cumsum.size - 1)

        mu = np.diff(cumsum[bin_idx])
        return (bins[:-1] + bins[1:]) / 2.0, np.maximum(mu, 0.0)

    def _bin_counts(self, ideal_rate_cps, dt_sec, duration, Tint_s, rng):
        bin_centers, mu_sig = self._bin_expected_counts(ideal_rate_cps, dt_sec, duration, Tint_s)
        if mu_sig.size == 0:
            return bin_centers, mu_sig

        mu_bkg = bkg_mu_series(
            n_bins=mu_sig.size,
            Tint_s=Tint_s,
            bkg_rate_cps=self.bkg_rate_cps_var.get(),
            bkg_drift_rsd_percent=self.bkg_drift_rsd_var.get(),
            bkg_corr_ms=self.bkg_corr_ms_var.get(),
            rng=rng,
        )
        obs = rng.poisson(mu_sig + mu_bkg).astype(float)
        obs = apply_gate_clustered(
            obs_counts=obs,
            mu_bkg=mu_bkg,
            gate_mode=self.bkg_gate_mode_var.get(),
            sigma_mult=self.bkg_sigma_mult_var.get(),
            min_run_bins=self.bkg_min_run_bins_var.get(),
        )
        return bin_centers, obs

    def _interval_sums_from_bins(self, counts_bins, Tint_s, t_starts, t_width):
        counts_bins = np.asarray(counts_bins, dtype=float)
        t_starts = np.asarray(t_starts, dtype=float)

        if counts_bins.size == 0 or Tint_s <= 0 or t_width <= 0 or t_starts.size == 0:
            return np.array([], dtype=float)

        n_bins = counts_bins.size
        bin_edges = np.arange(n_bins + 1, dtype=float) * Tint_s
        out = np.zeros(t_starts.size, dtype=float)

        for j, t0 in enumerate(t_starts):
            t1 = t0 + t_width
            k0 = max(0, int(np.floor(t0 / Tint_s)) - 1)
            k1 = min(n_bins - 1, int(np.floor(t1 / Tint_s)) + 1)

            total = 0.0
            for k in range(k0, k1 + 1):
                b0 = bin_edges[k]
                b1 = bin_edges[k + 1]
                overlap = max(0.0, min(t1, b1) - max(t0, b0))
                if overlap > 0:
                    total += counts_bins[k] * (overlap / Tint_s)
            out[j] = total

        return out

    def _counts_per_shot_from_bins(self, counts_bins, Tint_s, rep_hz, duration):
        if rep_hz <= 0 or Tint_s <= 0 or duration <= 0 or counts_bins.size == 0:
            return np.array([])
        Tshot = 1.0 / rep_hz
        n_shots = int(np.floor(duration / Tshot))
        if n_shots <= 0:
            return np.array([])
        return self._interval_sums_from_bins(counts_bins, Tint_s, np.arange(n_shots, dtype=float) * Tshot, Tshot)

    def _pixel_sums_from_bins(self, counts_bins, Tint_s, rep_hz, dose, start_pix, end_pix):
        n_pix = end_pix - start_pix
        if n_pix <= 0:
            return np.array([])
        T_pix = dose / max(rep_hz, 1e-12)
        return self._interval_sums_from_bins(
            counts_bins, Tint_s, (start_pix + np.arange(n_pix, dtype=float)) * T_pix, T_pix
        )

    def _pixel_expected_arrays(self, ideal_rate_cps, dt_sec, duration, Tint_s, rep_hz, dose, start_pix, end_pix):
        _, mu_bins = self._bin_expected_counts(ideal_rate_cps, dt_sec, duration, Tint_s)
        if mu_bins.size == 0:
            return np.array([]), np.array([]), np.array([])

        n_bins = mu_bins.size
        bin_edges = np.arange(n_bins + 1, dtype=float) * Tint_s
        T_pix = dose / max(rep_hz, 1e-12)
        n_pix = end_pix - start_pix
        if n_pix <= 0:
            return np.array([]), np.array([]), np.array([])

        means = np.zeros(n_pix, float)
        vars_ = np.zeros(n_pix, float)
        pix_idx = np.arange(start_pix, end_pix, dtype=int)

        for j in range(n_pix):
            t0 = (start_pix + j) * T_pix
            t1 = (start_pix + j + 1) * T_pix

            k0 = max(0, int(np.floor(t0 / Tint_s)) - 1)
            k1 = min(n_bins - 1, int(np.floor(t1 / Tint_s)) + 1)

            for k in range(k0, k1 + 1):
                b0 = bin_edges[k]
                b1 = bin_edges[k + 1]
                overlap = max(0.0, min(t1, b1) - max(t0, b0))
                if overlap > 0:
                    w = overlap / Tint_s
                    mu = float(mu_bins[k])
                    means[j] += w * mu
                    vars_[j] += (w * w) * mu

        return pix_idx, means, vars_

    def _pixel_expected_mean_var(self, ideal_rate_cps, dt_sec, duration, Tint_s, rep_hz, dose, start_pix, end_pix):
        _, means, vars_ = self._pixel_expected_arrays(
            ideal_rate_cps, dt_sec, duration, Tint_s, rep_hz, dose, start_pix, end_pix
        )
        if means.size == 0:
            return np.nan, np.nan, np.nan

        mean_mean = float(np.mean(means))
        mean_var = float(np.mean(vars_))
        if not (np.isfinite(mean_mean) and np.isfinite(mean_var)) or mean_var <= 0:
            return mean_mean, mean_var, np.nan

        return mean_mean, mean_var, float((mean_mean * mean_mean) / mean_var)

    def _pixel_rsd_at_report_tint(
        self,
        rep_hz,
        wash_ms,
        dose,
        start_pix_req,
        end_pix_req,
        class_name,
        shape_override=None,
    ):
        duration = float(self.duration_var.get())
        Tint_s = self._report_tint_s()

        rng_sig = np.random.default_rng(
            self._scenario_seed(9001, rep_hz, wash_ms, dose, len(class_name), hash(str(shape_override)) % 1000)
        )
        _, ideal_rate_cps, dt_sec, _ = self._ideal_signal_highres(
            washout_override_ms=wash_ms,
            freq_override_hz=rep_hz,
            class_override=class_name,
            rng=rng_sig,
            shape_override=shape_override,
        )

        T_pix = dose / max(rep_hz, 1e-9)
        n_pix_total = int(np.floor(duration / T_pix))
        start_pix, end_pix = clamp_pixel_window(start_pix_req, end_pix_req, n_pix_total, min_pixels=3)
        if start_pix is None:
            return np.nan

        rng2 = np.random.default_rng(
            self._scenario_seed(81000, rep_hz, wash_ms, dose, Tint_s * 1e6, len(class_name), hash(str(shape_override)) % 1000)
        )
        _, cb = self._bin_counts(ideal_rate_cps, dt_sec, duration, Tint_s, rng2)

        pix_vals = self._pixel_sums_from_bins(
            counts_bins=cb.astype(float),
            Tint_s=Tint_s,
            rep_hz=rep_hz,
            dose=dose,
            start_pix=start_pix,
            end_pix=end_pix,
        )
        return rsd_percent(pix_vals)

    def _background_counts_per_pixel(self, rep_hz, dose):
        pixel_time_s = dose / max(rep_hz, 1e-12)
        return self._bkg_cps() * pixel_time_s

    def _bg_limited_rsd_curve(self, x_signal_counts, b_counts):
        x = np.asarray(x_signal_counts, dtype=float)
        y = np.full_like(x, np.nan, dtype=float)
        good = x > 0
        y[good] = 100.0 * np.sqrt(x[good] + max(b_counts, 0.0)) / x[good]
        return y

    def _lod_counts(self, b_counts):
        return max(0.0, float(self.lod_sigma_var.get())) * np.sqrt(max(b_counts, 0.0))

    # --------------------------------------------------------
    # plots
    # --------------------------------------------------------
    def update_fast(self, *args):
        try:
            freq = float(self.freq_var.get())
            duration = float(self.duration_var.get())
            start_pix_req = int(self.start_pix_var.get())
            end_pix_req = int(self.end_pix_var.get())
            dose_stats = int(self.dose_var.get())
            current_shape = self._shape_name()
            active_class = self.active_class_var.get()
            signal_cps = self.current_signal_cps(class_name=active_class, rep_hz=freq)
            counts_per_shot_mean = self.current_counts_per_shot(class_name=active_class)
            mode_txt = "manual" if str(self.signal_mode_var.get()) == "Manual cts/shot" else "class"

            # ---------------- Signal ----------------
            rng_sig = np.random.default_rng(
                self._scenario_seed(
                    101,
                    freq,
                    self.width_var.get(),
                    signal_cps,
                    self.shot_yield_rsd_var.get(),
                    self.transport_rsd_var.get()
                )
            )
            _, ideal_rate_cps, dt_sec, _ = self._ideal_signal_highres(
                rng=rng_sig,
                class_override=active_class,
            )

            Tint_sig_s = float(self.int_time_var.get()) / 1000.0
            rng_bin = np.random.default_rng(
                self._scenario_seed(202, Tint_sig_s * 1e6, freq, self.width_var.get(), self.bkg_drift_rsd_var.get())
            )
            t_sig, c_sig = self._bin_counts(ideal_rate_cps, dt_sec, duration, Tint_sig_s, rng_bin)

            self.ax_sig.clear()
            if t_sig.size > 5000:
                self.ax_sig.plot(t_sig, c_sig, linewidth=0.8, color=CLASS_COLORS.get(active_class, "C1"))
            else:
                self.ax_sig.plot(t_sig, c_sig, drawstyle="steps-mid", linewidth=1.2, color=CLASS_COLORS.get(active_class, "C1"))
                self.ax_sig.fill_between(t_sig, c_sig, step="mid", alpha=0.1, color=CLASS_COLORS.get(active_class, "C1"))

            gate_mode = int(self.bkg_gate_mode_var.get())
            gate_txt = {0: "Gate:Off", 1: "Gate", 2: "Gate+Sub"}.get(gate_mode, "Gate:?")

            self.ax_sig.set_ylabel("Counts per integration")
            self.ax_sig.set_xlabel("Acquisition Time (s)")
            self.ax_sig.set_title(
                f"Signal | {active_class} | {self.current_class_conc_ppm(active_class):g} ppm | "
                f"{signal_cps:,.0f} cps | {counts_per_shot_mean:.2f} counts/shot | "
                f"mode={mode_txt} | shape={current_shape} | bkg={float(self.bkg_rate_cps_var.get()):g} cps | {gate_txt}"
            )
            self.ax_sig.grid(True, linestyle="--", alpha=0.5)
            self.ax_sig.set_xlim(0, duration)

            if c_sig.size > 0:
                mean_val = mean_by_mode(c_sig)
                max_val = float(np.max(c_sig))
                rsd = rsd_percent(c_sig)
                total = float(np.sum(c_sig))
                self.stat_mean.set(f"{mean_val:.2f}" if np.isfinite(mean_val) else "--")
                self.stat_max.set(f"{max_val:.0f}")
                self.stat_rsd.set(f"{rsd:.2f}%" if np.isfinite(rsd) else "--")
                self.stat_total.set(f"{total:.0f}")
            else:
                self.stat_mean.set("--")
                self.stat_max.set("--")
                self.stat_rsd.set("--")
                self.stat_total.set("--")

            shot_vals = self._counts_per_shot_from_bins(c_sig.astype(float), Tint_sig_s, freq, duration)
            if shot_vals.size >= 1:
                smean = mean_by_mode(shot_vals)
                srsd = rsd_percent(shot_vals)
                self.stat_shot_mean.set(f"{smean:.2f}" if np.isfinite(smean) else "--")
                self.stat_shot_rsd.set(f"{srsd:.2f}%" if np.isfinite(srsd) else "--")
            else:
                self.stat_shot_mean.set("--")
                self.stat_shot_rsd.set("--")

            T_pix = dose_stats / max(freq, 1e-9)
            n_pix_total = int(np.floor(duration / T_pix))
            sp, ep = clamp_pixel_window(start_pix_req, end_pix_req, n_pix_total, min_pixels=3)

            if sp is not None:
                pix_vals = self._pixel_sums_from_bins(c_sig.astype(float), Tint_sig_s, freq, dose_stats, sp, ep)
                pmean = mean_by_mode(pix_vals)
                prsd = rsd_percent(pix_vals)
                self.stat_pix_mean.set(f"{pmean:.2f}" if np.isfinite(pmean) else "--")
                self.stat_pix_rsd.set(f"{prsd:.2f}%" if np.isfinite(prsd) else "--")
            else:
                self.stat_pix_mean.set("--")
                self.stat_pix_rsd.set("--")

            self.export_frames["Signal"] = pd.DataFrame({
                "time_s": t_sig,
                "counts_per_integration": c_sig,
                "integration_ms": float(self.int_time_var.get()),
                "freq_hz": freq,
                "washout_ms": float(self.width_var.get()),
                "dose_shots_per_pixel": dose_stats,
                "shape": current_shape,
                "class": active_class,
                "concentration_ppm": self.concentration_map()[active_class],
                "signal_cps": signal_cps,
                "counts_per_shot_mean": counts_per_shot_mean,
                "spot_um": float(self.spot_um_var.get()),
                "depth_per_shot_um": float(self.depth_per_shot_um_var.get()),
                "mode": mode_txt,
            })

            tight_layout_for_right_legends(self.fig_sig, right=0.96, pad=2.0)
            self.canvas_sig.draw()

            # ---------------- Pixel RSD vs Tint ----------------
            self.ax_rsd.clear()
            Tint_values_s = np.logspace(np.log10(TINT_MIN_MS / 1000.0), np.log10(TINT_MAX_MS / 1000.0), FAST_TINT_N)
            rows_rsd = []

            for class_name in CONC_CLASSES:
                for wash_ms in self.overlay_washouts_ms:
                    rsd_curve = []

                    rng_sig_w = np.random.default_rng(
                        self._scenario_seed(
                            303,
                            freq,
                            wash_ms,
                            self.current_signal_cps(class_name=class_name, rep_hz=freq),
                            self.shot_yield_rsd_var.get(),
                            self.transport_rsd_var.get(),
                            hash(current_shape) % 1000
                        )
                    )
                    _, ideal_rate_w, dt_sec_w, _ = self._ideal_signal_highres(
                        washout_override_ms=wash_ms,
                        class_override=class_name,
                        rng=rng_sig_w,
                    )

                    dose = dose_stats
                    T_pix = dose / max(freq, 1e-9)
                    n_pix_total = int(np.floor(duration / T_pix))
                    sp, ep = clamp_pixel_window(start_pix_req, end_pix_req, n_pix_total, min_pixels=3)
                    if sp is None:
                        continue

                    for Tint_s in Tint_values_s:
                        rng2 = np.random.default_rng(
                            self._scenario_seed(
                                404,
                                freq,
                                wash_ms,
                                dose,
                                Tint_s * 1e6,
                                self.bkg_drift_rsd_var.get(),
                                len(class_name),
                                hash(current_shape) % 1000
                            )
                        )
                        _, cb = self._bin_counts(ideal_rate_w, dt_sec_w, duration, Tint_s, rng2)
                        pv = self._pixel_sums_from_bins(cb.astype(float), Tint_s, freq, dose, sp, ep)
                        rsd_val = rsd_percent(pv)
                        rsd_curve.append(rsd_val)
                        rows_rsd.append({
                            "tint_ms": Tint_s * 1e3,
                            "pixel_to_pixel_rsd_percent": rsd_val,
                            "dose_shots_per_pixel": dose,
                            "washout_ms": wash_ms,
                            "freq_hz": freq,
                            "shape": current_shape,
                            "class": class_name,
                            "concentration_ppm": self.concentration_map()[class_name],
                            "signal_cps": self.current_signal_cps(class_name=class_name, rep_hz=freq),
                        })

                    self.ax_rsd.plot(
                        Tint_values_s * 1e3,
                        rsd_curve,
                        linewidth=1.4,
                        color=CLASS_COLORS.get(class_name, "k"),
                        linestyle=WASH_LINESTYLES.get(wash_ms, "-"),
                        alpha=0.95,
                        label=f"{class_name} | wash={wash_ms:g} ms"
                    )

            self.ax_rsd.set_xscale("log")
            self.ax_rsd.grid(True, which="major", alpha=0.35)
            self.ax_rsd.grid(True, which="minor", alpha=0.15)
            self.ax_rsd.set_xlabel("Integration time Tint (ms)")
            self.ax_rsd.set_ylabel("Pixel-to-pixel RSD (%)")
            self.ax_rsd.set_title(f"Pixel sum RSD vs Tint | dose={dose_stats} | shape={current_shape}")

            class_handles = [Line2D([0], [0], color=CLASS_COLORS.get(cn, "k"), lw=2.0, label=cn) for cn in CONC_CLASSES]
            wash_handles = [Line2D([0], [0], color="black", lw=2.0, linestyle=WASH_LINESTYLES.get(w, "-"), label=f"wash={w:g} ms")
                            for w in self.overlay_washouts_ms]

            leg_class = legend_right_at(self.ax_rsd, y=0.72, handles=class_handles, title="Class (color)", fontsize=8, title_fontsize=8)
            self.ax_rsd.add_artist(leg_class)
            legend_right_at(self.ax_rsd, y=0.28, handles=wash_handles, title="Washout (linestyle)", fontsize=8, title_fontsize=8)

            self.export_frames["Pixel RSD vs Integration"] = pd.DataFrame(rows_rsd)

            tight_layout_for_right_legends(self.fig_rsd, right=0.78, pad=2.0)
            self.canvas_rsd.draw()

            # ---------------- Shape Comparison ----------------
            self.fig_shape.clf()
            gs_shape = self.fig_shape.add_gridspec(2, 1, height_ratios=[1.0, 1.2], hspace=0.28)
            ax_shape_top = self.fig_shape.add_subplot(gs_shape[0, 0])
            ax_shape_bot = self.fig_shape.add_subplot(gs_shape[1, 0])

            wash_ms = float(self.width_var.get())
            dose = int(self.dose_var.get())
            dt_shape = DT_US / 1e6
            rows_shape = []

            for shape_name in SHAPE_ORDER:
                tkern, kern = washout_kernel_same_model(wash_ms, dt_shape, shape_name)
                knorm = kern / np.max(kern) if np.max(kern) > 0 else kern
                ax_shape_top.plot(
                    tkern * 1e3,
                    knorm,
                    color=SHAPE_COLORS.get(shape_name, "k"),
                    linestyle=SHAPE_LINESTYLES.get(shape_name, "-"),
                    linewidth=1.6,
                    label=shape_name,
                )
                for t, y in zip(tkern * 1e3, knorm):
                    rows_shape.append({"panel": "normalized_shape", "shape": shape_name, "time_ms": t, "y": y})

            ax_shape_top.set_xlabel("Time after shot (ms)")
            ax_shape_top.set_ylabel("Normalized intensity")
            ax_shape_top.set_title(f"Transient shapes at washout = {wash_ms:g} ms")
            ax_shape_top.grid(True, alpha=0.25)
            legend_right_at(ax_shape_top, y=0.5, fontsize=8)

            Tint_values_s = np.logspace(np.log10(TINT_MIN_MS / 1000.0), np.log10(TINT_MAX_MS / 1000.0), FAST_TINT_N)
            active_class = self.active_class_var.get()

            for shape_name in SHAPE_ORDER:
                rng_sig_shape = np.random.default_rng(
                    self._scenario_seed(515, freq, wash_ms, dose, len(shape_name), len(active_class))
                )
                _, ideal_rate_shape, dt_shape2, _ = self._ideal_signal_highres(
                    washout_override_ms=wash_ms,
                    freq_override_hz=freq,
                    class_override=active_class,
                    rng=rng_sig_shape,
                    shape_override=shape_name,
                )

                T_pix = dose / max(freq, 1e-9)
                n_pix_total = int(np.floor(duration / T_pix))
                sp, ep = clamp_pixel_window(start_pix_req, end_pix_req, n_pix_total, min_pixels=3)
                if sp is None:
                    continue

                rsd_curve = []
                for Tint_s in Tint_values_s:
                    rng2 = np.random.default_rng(
                        self._scenario_seed(516, freq, wash_ms, dose, Tint_s * 1e6, len(shape_name))
                    )
                    _, cb = self._bin_counts(ideal_rate_shape, dt_shape2, duration, Tint_s, rng2)
                    pv = self._pixel_sums_from_bins(cb.astype(float), Tint_s, freq, dose, sp, ep)
                    rsd_val = rsd_percent(pv)
                    rsd_curve.append(rsd_val)
                    rows_shape.append({
                        "panel": "pixel_rsd_vs_tint",
                        "shape": shape_name,
                        "tint_ms": Tint_s * 1e3,
                        "pixel_to_pixel_rsd_percent": rsd_val,
                        "washout_ms": wash_ms,
                        "dose_shots_per_pixel": dose,
                        "freq_hz": freq,
                        "class": active_class,
                    })

                ax_shape_bot.plot(
                    Tint_values_s * 1e3,
                    rsd_curve,
                    color=SHAPE_COLORS.get(shape_name, "k"),
                    linestyle=SHAPE_LINESTYLES.get(shape_name, "-"),
                    linewidth=1.6,
                    label=shape_name,
                )

            ax_shape_bot.set_xscale("log")
            ax_shape_bot.set_xlabel("Integration time Tint (ms)")
            ax_shape_bot.set_ylabel("Pixel-to-pixel RSD (%)")
            ax_shape_bot.set_title(f"Shape sensitivity at dose={dose}, freq={freq:.0f} Hz, class={active_class}")
            ax_shape_bot.grid(True, which="major", alpha=0.35)
            ax_shape_bot.grid(True, which="minor", alpha=0.15)
            legend_right_at(ax_shape_bot, y=0.5, fontsize=8)

            self.export_frames["Shape Comparison"] = pd.DataFrame(rows_shape)

            self.fig_shape.subplots_adjust(left=0.08, right=0.78, top=0.93, bottom=0.10, hspace=0.35)
            self.canvas_shape.draw()

            # ---------------- Snapshot ----------------
            self.fig_snap.clf()
            gs = self.fig_snap.add_gridspec(2, 1, height_ratios=[1.1, 0.9], hspace=0.25)
            ax_top = self.fig_snap.add_subplot(gs[0, 0])
            ax_bot = self.fig_snap.add_subplot(gs[1, 0])

            rng_snap_sig = np.random.default_rng(
                self._scenario_seed(
                    505, freq, self.width_var.get(), signal_cps,
                    self.shot_yield_rsd_var.get(), self.transport_rsd_var.get(),
                    hash(current_shape) % 1000
                )
            )
            t_hi, sig_hi, dt_hi, _ = self._ideal_signal_highres(rng=rng_snap_sig, class_override=active_class)

            Tshot = 1.0 / max(freq, 1e-12)
            shot0 = int(self.snapshot_start_shot)
            nshots = int(self.snapshot_nshots)

            t0 = shot0 * Tshot
            t1 = min((shot0 + nshots) * Tshot, duration)
            t2 = (0.6)*(t1)
            t3 = t0 + ((0.095)*(t1))
            rows_snap = []

            mh = (t_hi >= t0) & (t_hi <= t1)
            if np.sum(mh) >= 2:
                ax_top.plot(t_hi[mh], sig_hi[mh], linewidth=1.0, alpha=0.65, label="Expected pulse train rate")
                for t, y in zip(t_hi[mh], sig_hi[mh]):
                    rows_snap.append({"panel": "expected_rate", "time_s": t, "value": y})

            tint_colors = {t: f"C{i}" for i, t in enumerate(self.snapshot_tints_ms)}
            for tint_ms in self.snapshot_tints_ms:
                Tint_s = tint_ms / 1000.0
                rng_snap = np.random.default_rng(
                    self._scenario_seed(606, Tint_s * 1e6, freq, self.width_var.get(), len(active_class))
                )
                t_bins, c_bins = self._bin_counts(sig_hi, dt_hi, duration, Tint_s, rng_snap)

                m = (t_bins >= t0) & (t_bins <= t1)
                if np.sum(m) < 2:
                    continue

                rate = c_bins.astype(float) / max(Tint_s, 1e-12)
                col = tint_colors.get(tint_ms, None)

                ax_top.plot(t_bins[m], rate[m], drawstyle="steps-mid", linewidth=1.4, color=col, label=f"Tint={tint_ms:g} ms (rate)")
                ax_bot.plot(t_bins[m], c_bins[m], marker="o", markersize=3, linewidth=0.8, color=col, label=f"Tint={tint_ms:g} ms")

                for t, y in zip(t_bins[m], rate[m]):
                    rows_snap.append({"panel": "rate_overlay", "tint_ms": tint_ms, "time_s": t, "value": y})
                for t, y in zip(t_bins[m], c_bins[m]):
                    rows_snap.append({"panel": "counts_overlay", "tint_ms": tint_ms, "time_s": t, "value": y})

            ax_top.set_title(f"Pulses (shots {shot0}–{shot0+nshots-1}) with Overlays")
            ax_top.set_ylabel("Counts per second")
            ax_top.set_xlim(t3, t2)
            ax_top.grid(True, alpha=0.25)

            ax_bot.set_title("Counts per datapoint")
            ax_bot.set_xlabel("Time (s)")
            ax_bot.set_ylabel("Counts per integration")
            ax_bot.set_xlim(t3, t2)
            ax_bot.grid(True, alpha=0.25)

            legend_right_at(ax_top, y=0.5, fontsize=8)
            legend_right_at(ax_bot, y=0.5, fontsize=8)

            self.export_frames["Integration Snapshot"] = pd.DataFrame(rows_snap)

            self.fig_snap.subplots_adjust(left=0.08, right=0.78, top=0.93, bottom=0.10, hspace=0.35)
            self.canvas_snap.draw()

            # ---------------- Snapshot + Threshold ----------------
            self.fig_snap_thresh.clf()
            gs_thr = self.fig_snap_thresh.add_gridspec(2, 1, height_ratios=[1.1, 0.9], hspace=0.25)
            ax_top_thr = self.fig_snap_thresh.add_subplot(gs_thr[0, 0])
            ax_bot_thr = self.fig_snap_thresh.add_subplot(gs_thr[1, 0])

            rng_snap_sig_thr = np.random.default_rng(
                self._scenario_seed(
                    705, freq, self.width_var.get(), signal_cps,
                    self.shot_yield_rsd_var.get(), self.transport_rsd_var.get(),
                    hash(current_shape) % 1000
                )
            )
            t_hi_thr, sig_hi_thr, dt_hi_thr, _ = self._ideal_signal_highres(rng=rng_snap_sig_thr, class_override=active_class)

            Tshot = 1.0 / max(freq, 1e-12)
            shot0 = int(self.snapshot_start_shot)
            nshots = int(self.snapshot_nshots)
            t0 = shot0 * Tshot
            t1 = min((shot0 + nshots) * Tshot, duration)
            rows_snap_thr = []

            mh = (t_hi_thr >= t0) & (t_hi_thr <= t1)
            if np.sum(mh) >= 2:
                ax_top_thr.plot(t_hi_thr[mh], sig_hi_thr[mh], linewidth=1.0, alpha=0.65, label="Expected pulse train rate")

            tint_colors = {t: f"C{i}" for i, t in enumerate(self.snapshot_tints_ms)}
            bkg_cps = self._bkg_cps()
            sigma_mult = float(self.lod_sigma_var.get())

            for tint_ms in self.snapshot_tints_ms:
                Tint_s = tint_ms / 1000.0
                rng_snap_thr = np.random.default_rng(
                    self._scenario_seed(706, Tint_s * 1e6, freq, self.width_var.get(), len(active_class))
                )
                t_bins, c_bins = self._bin_counts(sig_hi_thr, dt_hi_thr, duration, Tint_s, rng_snap_thr)

                rng_bkg_only = np.random.default_rng(
                    self._scenario_seed(716, Tint_s * 1e6, freq, self.width_var.get(), len(active_class))
                )

                m = (t_bins >= t0) & (t_bins <= t1)
                if np.sum(m) < 2:
                    continue

                rate = c_bins.astype(float) / max(Tint_s, 1e-12)
                col = tint_colors.get(tint_ms, None)

                mu_bkg = self._bkg_counts_for_tint(Tint_s)
                thresh_counts = mu_bkg + sigma_mult * np.sqrt(max(mu_bkg, 0.0))
                thresh_rate = thresh_counts / max(Tint_s, 1e-12)

                bkg_only_counts = rng_bkg_only.poisson(mu_bkg, size=len(t_bins)).astype(float)
                bkg_only_rate = bkg_only_counts / max(Tint_s, 1e-12)

                ax_top_thr.plot(t_bins[m], rate[m], drawstyle="steps-mid", linewidth=1.4, color=col, label=f"Tint={tint_ms:g} ms (rate)")
                ax_bot_thr.plot(t_bins[m], c_bins[m], marker="o", markersize=3, linewidth=0.8, color=col, label=f"Tint={tint_ms:g} ms")

                ax_top_thr.plot(t_bins[m], bkg_only_rate[m], color=col, linewidth=0.8, alpha=0.25)
                ax_bot_thr.plot(t_bins[m], bkg_only_counts[m], color=col, linewidth=0.8, alpha=0.25)

                ax_top_thr.axhline(bkg_cps, color=col, linestyle=":", linewidth=1.0, alpha=0.8)
                ax_top_thr.axhline(thresh_rate, color=col, linestyle="--", linewidth=1.0, alpha=0.8)
                ax_bot_thr.axhline(mu_bkg, color=col, linestyle=":", linewidth=1.0, alpha=0.8)
                ax_bot_thr.axhline(thresh_counts, color=col, linestyle="--", linewidth=1.0, alpha=0.8)

                rows_snap_thr.append({
                    "tint_ms": tint_ms,
                    "background_cps": bkg_cps,
                    "threshold_rate_cps": thresh_rate,
                    "background_counts_per_bin": mu_bkg,
                    "threshold_counts_per_bin": thresh_counts,
                })

            ax_top_thr.set_title(f"Pulses (shots {shot0}–{shot0+nshots-1}) with Background + Threshold")
            ax_top_thr.set_ylabel("Counts per second")
            ax_top_thr.set_xlim(t0, t1)
            ax_top_thr.grid(True, alpha=0.25)

            ax_bot_thr.set_title("Counts per datapoint with Background + Threshold")
            ax_bot_thr.set_xlabel("Time (s)")
            ax_bot_thr.set_ylabel("Counts per integration")
            ax_bot_thr.set_xlim(t0, t1)
            ax_bot_thr.grid(True, alpha=0.25)

            extra_handles = [
                Line2D([0], [0], color="black", linestyle=":", linewidth=1.2, label="Mean background"),
                Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label=f"{sigma_mult:g}σ threshold"),
                Line2D([0], [0], color="black", linestyle="-", linewidth=1.0, alpha=0.25, label="Noisy background realization"),
            ]

            leg_top_thr = legend_right_at(ax_top_thr, y=0.60, fontsize=8)
            if leg_top_thr:
                ax_top_thr.add_artist(leg_top_thr)
            legend_right_at(ax_top_thr, y=0.18, handles=extra_handles, title="Reference lines", fontsize=8, title_fontsize=8)

            leg_bot_thr = legend_right_at(ax_bot_thr, y=0.60, fontsize=8)
            if leg_bot_thr:
                ax_bot_thr.add_artist(leg_bot_thr)
            legend_right_at(ax_bot_thr, y=0.18, handles=extra_handles, title="Reference lines", fontsize=8, title_fontsize=8)

            self.export_frames["Integration Snapshot + Threshold"] = pd.DataFrame(rows_snap_thr)

            self.fig_snap_thresh.subplots_adjust(left=0.08, right=0.78, top=0.93, bottom=0.10, hspace=0.35)
            self.canvas_snap_thresh.draw()

        except Exception as e:
            print("ERROR (update_fast):", e)

    def update_full(self, *args):
        try:
            self.update_fast()

            freq = float(self.freq_var.get())
            duration = float(self.duration_var.get())
            start_pix_req = int(self.start_pix_var.get())
            end_pix_req = int(self.end_pix_var.get())
            Tint_ref_s = self._report_tint_s()
            report_tint_ms = float(self.report_tint_ms_var.get())
            current_shape = self._shape_name()

            # ---------------- Pixel-to-Pixel Precision ----------------
            self.ax_master.clear()
            master_rows = []

            for rep_hz in self.rep_rates:
                for wash_ms in self.overlay_washouts_ms:
                    for dose in self.overlay_doses:
                        b_counts = self._background_counts_per_pixel(rep_hz, dose)

                        for class_name in CONC_CLASSES:
                            baseline_rsd = self._pixel_rsd_at_report_tint(
                                rep_hz, wash_ms, dose, start_pix_req, end_pix_req,
                                class_name=class_name, shape_override=current_shape,
                            )
                            if not np.isfinite(baseline_rsd):
                                continue

                            rng_sig = np.random.default_rng(
                                self._scenario_seed(
                                    707, rep_hz, wash_ms, dose,
                                    self.current_signal_cps(class_name=class_name, rep_hz=rep_hz),
                                    self.shot_yield_rsd_var.get(), self.transport_rsd_var.get(),
                                    hash(current_shape) % 1000
                                )
                            )
                            _, ideal_sig_c, dt_c, _ = self._ideal_signal_highres(
                                washout_override_ms=wash_ms,
                                freq_override_hz=rep_hz,
                                class_override=class_name,
                                rng=rng_sig,
                                shape_override=current_shape,
                            )

                            T_pix = dose / max(rep_hz, 1e-12)
                            n_pix_total = int(np.floor(duration / T_pix))
                            sp, ep = clamp_pixel_window(start_pix_req, end_pix_req, n_pix_total, min_pixels=3)
                            if sp is None:
                                continue

                            mean_pix, var_pix, Neff = self._pixel_expected_mean_var(
                                ideal_sig_c, dt_c, duration, Tint_ref_s, rep_hz, dose, sp, ep
                            )

                            master_rows.append((mean_pix, var_pix, Neff, baseline_rsd, rep_hz, wash_ms, dose, class_name, b_counts))

            rows_master = []
            if len(master_rows) < 5:
                self.ax_master.text(0.5, 0.5, "Not enough rows.", transform=self.ax_master.transAxes, ha="center", va="center")
            else:
                xs_mean = []

                for (mean_pix, var_pix, Neff, brsd, rep_hz, wash_ms, dose, class_name, b_counts) in master_rows:
                    base = DOSE_COLORS.get(dose, "gray")
                    fc = shade_color(base, REP_SHADE.get(rep_hz, 1.0))

                    self.ax_master.scatter(
                        mean_pix, brsd, s=60,
                        marker=WASH_MARKERS.get(wash_ms, "o"),
                        facecolor=fc, edgecolor=EDGE_FOR_CLASS.get(class_name, "black"),
                        linewidth=0.7, alpha=0.9
                    )

                    rows_master.append({
                        "mean_counts_per_pixel": mean_pix,
                        "pixel_rsd_percent": brsd,
                        "rep_rate_hz": rep_hz,
                        "washout_ms": wash_ms,
                        "dose_shots_per_pixel": dose,
                        "class": class_name,
                        "concentration_ppm": self.concentration_map()[class_name],
                        "signal_cps": self.current_signal_cps(class_name=class_name, rep_hz=rep_hz),
                        "counts_per_shot": self.current_counts_per_shot(class_name=class_name),
                        "background_counts_per_pixel": b_counts,
                        "shape": current_shape,
                    })

                    if np.isfinite(mean_pix) and mean_pix > 0:
                        xs_mean.append(max(mean_pix, 1e-12))

                self.ax_master.set_xscale("log")
                self.ax_master.set_xlim(left=1e0)
                self.ax_master.set_xlabel("Mean counts per pixel (expected, noise-free)")
                self.ax_master.set_ylabel(f"Pixel RSD at {report_tint_ms:g} ms (%)")
                self.ax_master.set_title(f"Pixel RSD at {report_tint_ms:g} ms vs Pixel Counts | shape={current_shape}")
                self.ax_master.set_ylim(0, MASTER_RSD_YMAX)
                self.ax_master.grid(True, which="major", alpha=0.25)
                self.ax_master.grid(True, which="minor", alpha=0.12)

                if len(xs_mean) >= 1 and self.show_theory_var.get():
                    xgrid = np.logspace(0, 5, 400)
                    y_count = 100.0 / np.sqrt(np.maximum(xgrid, 1e-12))
                    self.ax_master.plot(xgrid, y_count, linewidth=1.2, linestyle="--", color="black", label="Pure counting floor")

                if self.show_lod_var.get() and len(master_rows) > 0:
                    b_med = float(np.nanmedian([row[8] for row in master_rows]))
                    xgrid = np.logspace(0, 5, 400)
                    y_bg = self._bg_limited_rsd_curve(xgrid, b_med)
                    self.ax_master.plot(xgrid, y_bg, linewidth=1.2, linestyle="-.", color="black", label="Background-limited precision")

                    lod_x = self._lod_counts(b_med)
                    if np.isfinite(lod_x) and lod_x > 0:
                        self.ax_master.axvline(lod_x, linewidth=1.0, linestyle="-", color="black")
                        self.ax_master.text(lod_x, MASTER_RSD_YMAX * 0.95, f"{self.lod_sigma_var.get():g}σ LOD", rotation=90, va="top", ha="right", fontsize=8)

                class_handles = [
                    Line2D([0], [0], marker="o", linestyle="none",
                           markerfacecolor="white", markeredgecolor=EDGE_FOR_CLASS.get(cn, "black"),
                           markeredgewidth=2.0, label=cn, markersize=8)
                    for cn in CONC_CLASSES
                ]
                dose_handles = [
                    Line2D([0], [0], marker="o", linestyle="none",
                           markerfacecolor=DOSE_COLORS.get(d, "gray"), markeredgecolor="black",
                           label=f"dose={d}", markersize=8)
                    for d in self.overlay_doses
                ]
                rep_handles = [
                    Line2D([0], [0], marker="o", linestyle="none",
                           markerfacecolor=shade_color("black", REP_SHADE.get(r, 1.0)),
                           markeredgecolor="black", label=f"{int(r)} Hz", markersize=8)
                    for r in self.rep_rates
                ]
                wash_handles = [
                    Line2D([0], [0], marker=WASH_MARKERS.get(w, "o"), linestyle="none",
                           markerfacecolor="white", markeredgecolor="black",
                           label=f"wash={w:g} ms", markersize=8)
                    for w in self.overlay_washouts_ms
                ]

                ref_handle = []
                if self.show_theory_var.get():
                    ref_handle.append(Line2D([0], [0], linestyle="--", color="black", label="Pure counting floor"))
                if self.show_lod_var.get():
                    ref_handle.append(Line2D([0], [0], linestyle="-.", color="black", label="Background-limited precision"))

                leg1 = legend_right_at(self.ax_master, y=0.82, handles=class_handles, title="Class (outline)", fontsize=8, title_fontsize=8)
                self.ax_master.add_artist(leg1)
                leg2 = legend_right_at(self.ax_master, y=0.58, handles=dose_handles, title="Dose (fill)", fontsize=8, title_fontsize=8)
                self.ax_master.add_artist(leg2)
                leg3 = legend_right_at(self.ax_master, y=0.34, handles=rep_handles, title="Rep-rate (shade)", fontsize=8, title_fontsize=8)
                self.ax_master.add_artist(leg3)
                legend_right_at(self.ax_master, y=0.10, handles=wash_handles + ref_handle, title="Washout / Theory", fontsize=8, title_fontsize=8)

            self.export_frames["Pixel-to-Pixel Precision"] = pd.DataFrame(rows_master)
            tight_layout_for_right_legends(self.fig_master, right=0.78, pad=2.0)
            self.canvas_master.draw()

            # ---------------- Single-Pixel Counting Precision ----------------
            self.ax_singlepix.clear()
            rows_single = []

            for rep_hz in self.rep_rates:
                for wash_ms in self.overlay_washouts_ms:
                    for dose in self.overlay_doses:
                        b_counts = self._background_counts_per_pixel(rep_hz, dose)

                        for class_name in CONC_CLASSES:
                            rng_sig = np.random.default_rng(
                                self._scenario_seed(
                                    1707, rep_hz, wash_ms, dose,
                                    self.current_signal_cps(class_name=class_name, rep_hz=rep_hz),
                                    self.shot_yield_rsd_var.get(), self.transport_rsd_var.get(),
                                    hash(current_shape) % 1000
                                )
                            )
                            _, ideal_sig_c, dt_c, _ = self._ideal_signal_highres(
                                washout_override_ms=wash_ms,
                                freq_override_hz=rep_hz,
                                class_override=class_name,
                                rng=rng_sig,
                                shape_override=current_shape,
                            )

                            T_pix = dose / max(rep_hz, 1e-12)
                            n_pix_total = int(np.floor(duration / T_pix))
                            sp, ep = clamp_pixel_window(start_pix_req, end_pix_req, n_pix_total, min_pixels=3)
                            if sp is None:
                                continue

                            mean_pix, _, _ = self._pixel_expected_mean_var(
                                ideal_sig_c, dt_c, duration, Tint_ref_s, rep_hz, dose, sp, ep
                            )
                            if not (np.isfinite(mean_pix) and mean_pix > 0):
                                continue

                            single_rsd = 100.0 * np.sqrt(mean_pix + b_counts) / mean_pix

                            base = DOSE_COLORS.get(dose, "gray")
                            fc = shade_color(base, REP_SHADE.get(rep_hz, 1.0))
                            self.ax_singlepix.scatter(
                                mean_pix, single_rsd, s=60,
                                marker=WASH_MARKERS.get(wash_ms, "o"),
                                facecolor=fc, edgecolor=EDGE_FOR_CLASS.get(class_name, "black"),
                                linewidth=0.7, alpha=0.9
                            )

                            rows_single.append({
                                "signal_counts_per_pixel": mean_pix,
                                "single_pixel_rsd_percent": single_rsd,
                                "background_counts_per_pixel": b_counts,
                                "rep_rate_hz": rep_hz,
                                "washout_ms": wash_ms,
                                "dose_shots_per_pixel": dose,
                                "class": class_name,
                                "concentration_ppm": self.concentration_map()[class_name],
                                "signal_cps": self.current_signal_cps(class_name=class_name, rep_hz=rep_hz),
                                "counts_per_shot": self.current_counts_per_shot(class_name=class_name),
                                "shape": current_shape,
                            })

            self.ax_singlepix.set_xscale("log")
            self.ax_singlepix.set_xlim(left=1e0)
            self.ax_singlepix.set_xlabel("Signal counts per pixel")
            self.ax_singlepix.set_ylabel("Single-pixel RSD (%)")
            self.ax_singlepix.set_title(f"Single-pixel counting/background precision | shape={current_shape}")
            self.ax_singlepix.set_ylim(0, SINGLE_PIXEL_RSD_YMAX)
            self.ax_singlepix.grid(True, which="major", alpha=0.25)
            self.ax_singlepix.grid(True, which="minor", alpha=0.12)

            if len(rows_single) > 0:
                b_med = float(np.nanmedian([r["background_counts_per_pixel"] for r in rows_single]))
                xgrid = np.logspace(0, 5, 400)

                if self.show_theory_var.get():
                    y_count = 100.0 / np.sqrt(np.maximum(xgrid, 1e-12))
                    self.ax_singlepix.plot(xgrid, y_count, linestyle="--", linewidth=1.2, color="black", label="Pure counting floor")

                if self.show_lod_var.get():
                    y_bg = self._bg_limited_rsd_curve(xgrid, b_med)
                    self.ax_singlepix.plot(xgrid, y_bg, linestyle="-.", linewidth=1.2, color="black", label="Background-limited precision")
                    lod_x = self._lod_counts(b_med)
                    if np.isfinite(lod_x) and lod_x > 0:
                        self.ax_singlepix.axvline(lod_x, linewidth=1.0, linestyle="-", color="black")
                        self.ax_singlepix.text(lod_x, SINGLE_PIXEL_RSD_YMAX * 0.95, f"{self.lod_sigma_var.get():g}σ LOD", rotation=90, va="top", ha="right", fontsize=8)

                class_handles = [
                    Line2D([0], [0], marker="o", linestyle="none",
                           markerfacecolor="white", markeredgecolor=EDGE_FOR_CLASS.get(cn, "black"),
                           markeredgewidth=2.0, label=cn, markersize=8)
                    for cn in CONC_CLASSES
                ]
                dose_handles = [
                    Line2D([0], [0], marker="o", linestyle="none",
                           markerfacecolor=DOSE_COLORS.get(d, "gray"), markeredgecolor="black",
                           label=f"dose={d}", markersize=8)
                    for d in self.overlay_doses
                ]
                rep_handles = [
                    Line2D([0], [0], marker="o", linestyle="none",
                           markerfacecolor=shade_color("black", REP_SHADE.get(r, 1.0)),
                           markeredgecolor="black", label=f"{int(r)} Hz", markersize=8)
                    for r in self.rep_rates
                ]
                wash_handles = [
                    Line2D([0], [0], marker=WASH_MARKERS.get(w, "o"), linestyle="none",
                           markerfacecolor="white", markeredgecolor="black",
                           label=f"wash={w:g} ms", markersize=8)
                    for w in self.overlay_washouts_ms
                ]

                theory_handles = []
                if self.show_theory_var.get():
                    theory_handles.append(Line2D([0], [0], linestyle="--", color="black", label="Pure counting floor"))
                if self.show_lod_var.get():
                    theory_handles.append(Line2D([0], [0], linestyle="-.", color="black", label="Background-limited precision"))

                leg1 = legend_right_at(self.ax_singlepix, y=0.82, handles=class_handles, title="Class (outline)", fontsize=8, title_fontsize=8)
                self.ax_singlepix.add_artist(leg1)
                leg2 = legend_right_at(self.ax_singlepix, y=0.58, handles=dose_handles, title="Dose (fill)", fontsize=8, title_fontsize=8)
                self.ax_singlepix.add_artist(leg2)
                leg3 = legend_right_at(self.ax_singlepix, y=0.34, handles=rep_handles, title="Rep-rate (shade)", fontsize=8, title_fontsize=8)
                self.ax_singlepix.add_artist(leg3)
                legend_right_at(self.ax_singlepix, y=0.10, handles=wash_handles + theory_handles, title="Washout / Theory", fontsize=8, title_fontsize=8)

            self.export_frames["Single-Pixel Counting Precision"] = pd.DataFrame(rows_single)
            tight_layout_for_right_legends(self.fig_singlepix, right=0.78, pad=2.0)
            self.canvas_singlepix.draw()

        except Exception as e:
            print("ERROR (update_full):", e)


if __name__ == "__main__":
    root = tk.Tk()
    app = LaTofSimulator(root)
    root.mainloop()
