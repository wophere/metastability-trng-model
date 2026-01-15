#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-figure comparison of conditional min-entropy H_min(n):
  - rho = 0 (independent) baseline
  - rho != 0 first-order (linear) approximation
  - rho != 0 Monte Carlo (MC) estimate

Hmin(n) = -log2( max(p_cond(n), 1 - p_cond(n)) )

We use the "all-same-sign" pattern (often the most biased under positive correlation):
  b_0 = ... = b_{n-1} = b_n = b,  b in {+1,-1}

First-order approximation:
  if r != 0:
    p_cond ≈ Φ(b r) + (φ(r)^2 / Φ(b r)) * Σ_{k=1}^n ρ[k]
  if r == 0:
    p_cond ≈ 1/2 + (1/π) * Σ_{k=1}^n ρ[k]

Correlation model:
  ρ[k] = alpha * k^{-beta}, 0 < beta < 1
  alpha = σ_f^2 / (σ_w^2 + σ_f^2)

Notes:
- The first-order approximation can leave [0,1] for large n or large alpha; we clip.
- Monte Carlo requires the Toeplitz correlation matrix to be PSD. We add a tiny diagonal jitter if needed.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------- standard normal pdf/cdf (no scipy needed) ----------
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)

def phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT2PI

def Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


# ---------- correlation model ----------
def rho_lags(n: int, alpha: float, beta: float) -> np.ndarray:
    """Return rho[1..n] as numpy array length n."""
    k = np.arange(1, n + 1, dtype=float)
    return alpha * np.power(k, -beta)


# ---------- min-entropy ----------
def hmin_from_p(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    q = max(p, 1.0 - p)
    return -math.log(q, 2)


# ---------- first-order p_cond (all-same pattern) ----------
def p_cond_first_order_all_same(n: int, r: float, alpha: float, beta: float, b: int = 1) -> float:
    """
    First-order approximation of p_cond for:
      b_0=...=b_{n-1}=b_n=b
    """
    rho = rho_lags(n, alpha=alpha, beta=beta)  # rho[1..n]
    s1 = float(np.sum(rho))

    if abs(r) < 1e-14:
        p = 0.5 + (1.0 / math.pi) * s1
    else:
        p0 = Phi(b * r)
        p0 = min(max(p0, 1e-15), 1.0 - 1e-15)
        p = p0 + (phi(r) ** 2 / p0) * s1

    return float(np.clip(p, 1e-12, 1.0 - 1e-12))


# ---------- Toeplitz correlation + PSD jitter ----------
def toeplitz_corr_matrix(n: int, alpha: float, beta: float) -> np.ndarray:
    """
    Build (n+1)x(n+1) Toeplitz correlation matrix for lags 0..n.
    """
    dim = n + 1
    rho = rho_lags(n, alpha=alpha, beta=beta)  # rho[1..n]
    first_row = np.concatenate(([1.0], rho))
    R = np.empty((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            R[i, j] = first_row[abs(i - j)]
    return R

def cholesky_with_jitter(R: np.ndarray, jitter0: float = 1e-12, max_tries: int = 12) -> np.ndarray:
    """
    Try Cholesky; if fails, add diagonal jitter (increasing) until it works.
    Returns lower-triangular L such that (R + jitter*I) = L L^T.
    """
    jitter = jitter0
    I = np.eye(R.shape[0], dtype=float)
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(R + jitter * I)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError("Cholesky failed even after adding diagonal jitter.")


# ---------- Monte Carlo estimate of p_cond (all-same pattern) ----------
def p_cond_monte_carlo_all_same(
    n: int,
    r: float,
    alpha: float,
    beta: float,
    b: int = 1,
    num_samples: int = 200_000,
    seed: int = 0,
) -> float:
    """
    Estimate p_cond = P(B_n=b | B_0..B_{n-1}=b) by Monte Carlo.

    Sample V ~ N(mu=r, Sigma=R) with sigma=1. Use sign(V) as bits.
    """
    rng = np.random.default_rng(seed)
    dim = n + 1

    R = toeplitz_corr_matrix(n, alpha=alpha, beta=beta)
    L = cholesky_with_jitter(R)

    eps = rng.standard_normal(size=(dim, num_samples))
    Z = (L @ eps).T  # (num_samples, dim)
    V = Z + r

    B = np.where(V >= 0.0, 1, -1)
    past_ok = np.all(B[:, :n] == b, axis=1)
    denom = int(np.sum(past_ok))
    if denom == 0:
        return float("nan")

    num = int(np.sum(B[past_ok, n] == b))
    p = num / denom
    return float(np.clip(p, 1e-12, 1.0 - 1e-12))

def plot_hmin_two_panel(
    n_max: int = 12,
    beta: float = 0.05,
    sigma_w: float = 1.0,
    sigma_f_list=(0.01, 0.1, 0.3),
    r_left: float = 0.2,
    r_list=(0.0, 0.1, 0.3, 0.7),
    sigma_f_right: float = 0.1,
    b: int = 1,
    mc_samples: int = 300_000,
    seed: int = 1234,
    mc_ns=None,   # e.g. mc_ns = np.arange(1, n_max+1) or a subset (n=0 is always shown as baseline)
):
    """
    Two-panel figure with x-axis n = 0..n_max (shared y-axis):
      Left : sweep sigma_f at fixed r=r_left
      Right: sweep r at fixed sigma_f=sigma_f_right

    Baseline:
      - The former "rho=0" value is treated as the n=0 (unconditional) point.

    Overlays for n>=1:
      - Linear approximation (solid)
      - Monte Carlo estimate (markers)
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # ---------- Style: larger fonts / thicker lines / larger markers ----------
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })
    LW_LIN = 3.0
    LW_MC  = 2.2
    MS_MC  = 7.0
    MS_B0  = 8.0

    ns = np.arange(0, n_max + 1)          # include n=0
    ns_pos = np.arange(1, n_max + 1)      # n>=1 part

    if mc_ns is None:
        mc_ns = ns_pos
    mc_ns_set = set(int(x) for x in np.asarray(mc_ns).astype(int))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)

    # ---------------- Left subplot: vary sigma_f, fixed r ----------------
    r = r_left
    H0_left = hmin_from_p(Phi(b * r))  # n=0 baseline

    for idx, sigma_f in enumerate(sigma_f_list):
        color = colors[idx % len(colors)]
        alpha = (sigma_f**2) / (sigma_w**2 + sigma_f**2)

        # linear: prepend n=0 baseline
        H_lin_pos = np.array([
            hmin_from_p(p_cond_first_order_all_same(int(n), r=r, alpha=alpha, beta=beta, b=b))
            for n in ns_pos
        ], dtype=float)
        H_lin = np.concatenate(([H0_left], H_lin_pos))

        # MC: show markers for selected n in mc_ns_set (n>=1); keep n=0 as baseline marker (once)
        H_mc = np.full_like(ns, np.nan, dtype=float)
        for i, n in enumerate(ns_pos, start=1):
            if int(n) not in mc_ns_set:
                continue
            p_mc = p_cond_monte_carlo_all_same(
                n=int(n), r=r, alpha=alpha, beta=beta, b=b,
                num_samples=mc_samples, seed=seed + 1000 * idx + int(n)
            )
            if not math.isnan(p_mc):
                H_mc[i] = hmin_from_p(p_mc)

        # plot
        axL.plot(ns, H_lin, color=color, linewidth=LW_LIN)
        axL.plot(ns, H_mc,  linestyle="--", color=color, marker="o", markersize=MS_MC, linewidth=LW_MC)

    # single baseline marker at n=0 (avoid overplot since it's identical for all sigma_f)
    axL.plot([0], [H0_left], marker="D", linestyle="None", color="black", markersize=MS_B0, zorder=5)

    axL.set_title(rf"Effect of $r_f$ (fixed $r_u={r_left:g}$, $\beta={beta:g}$)")
    axL.set_xlabel(r"History length $n$")
    axL.set_ylabel("Conditional Entropy ")
    axL.grid(True, linewidth=1.0, alpha=0.35)
    axL.set_ylim(0.6, 1)
    axL.set_xlim(0, n_max)
    axL.set_xticks(np.arange(0, n_max + 1, 2))

    method_handles_L = [
        Line2D([0], [0], color="black", marker="D", linestyle="None",
               markersize=MS_B0, label=r"i.i.d. baseline ($n=0$)"),
        Line2D([0], [0], color="black", linestyle="-", lw=LW_LIN, label="Linear approximation"),
        Line2D([0], [0], color="black", marker="o", linestyle="None",
               markersize=MS_MC, label="Monte Carlo"),
    ]
    param_handles_L = [
        Line2D([0], [0], color=colors[i % len(colors)], lw=LW_LIN,
               label=rf"$r_f$={sf:g}")
        for i, sf in enumerate(sigma_f_list)
    ]
    leg1 = axL.legend(handles=method_handles_L, loc="lower left", frameon=True)
    axL.add_artist(leg1)
    axL.legend(handles=param_handles_L, loc="lower right", frameon=True, title="Colors")

    # ---------------- Right subplot: vary r, fixed sigma_f ----------------
    sigma_f = sigma_f_right
    alpha_right = (sigma_f**2) / (sigma_w**2 + sigma_f**2)

    for idx, r in enumerate(r_list):
        color = colors[idx % len(colors)]
        H0 = hmin_from_p(Phi(b * r))  # n=0 baseline (depends on r)

        # linear with n=0 prepended
        H_lin_pos = np.array([
            hmin_from_p(p_cond_first_order_all_same(int(n), r=r, alpha=alpha_right, beta=beta, b=b))
            for n in ns_pos
        ], dtype=float)
        H_lin = np.concatenate(([H0], H_lin_pos))

        # MC (n>=1)
        H_mc = np.full_like(ns, np.nan, dtype=float)
        for i, n in enumerate(ns_pos, start=1):
            if int(n) not in mc_ns_set:
                continue
            p_mc = p_cond_monte_carlo_all_same(
                n=int(n), r=r, alpha=alpha_right, beta=beta, b=b,
                num_samples=mc_samples, seed=seed + 2000 * idx + int(n)
            )
            if not math.isnan(p_mc):
                H_mc[i] = hmin_from_p(p_mc)

        axR.plot(ns, H_lin, color=color, linewidth=LW_LIN)
        axR.plot(ns, H_mc,  linestyle="--", color=color, marker="o", markersize=MS_MC, linewidth=LW_MC)
        axR.plot([0], [H0], linestyle="--", marker="D", color=color, markersize=MS_B0, zorder=5)

    axR.set_title(rf"Effect of $r_u$ (fixed $r_f={sigma_f_right:g}$, $\beta={beta:g}$)")
    axR.set_xlabel(r"History length $n$")
    axR.grid(True, linewidth=1.0, alpha=0.35)
    axR.set_xlim(0, n_max)
    axR.set_xticks(np.arange(0, n_max + 1, 2))
    axR.set_ylim(0.3, 1)
    method_handles_R = [
        Line2D([0], [0], color="black", marker="D", linestyle="None",
               markersize=MS_B0, label=r"i.i.d. baseline ($n=0$)"),
        Line2D([0], [0], color="black", linestyle="-", lw=LW_LIN, label="Linear approximation"),
        Line2D([0], [0], color="black", marker="o", linestyle="None",
               markersize=MS_MC, label="Monte Carlo"),
    ]
    param_handles_R = [
        Line2D([0], [0], color=colors[i % len(colors)], lw=LW_LIN,
               label=rf"$r_u$={rv:g}")
        for i, rv in enumerate(r_list)
    ]
    leg2 = axR.legend(handles=method_handles_R, loc="lower left", frameon=True)
    axR.add_artist(leg2)
    axR.legend(handles=param_handles_R, loc="lower right", frameon=True, title="Colors")

    # fig.suptitle(
    #     r"Conditional Entropy versus History Length $n$ (Linear approximation and Monte Carlo)",
    #     y=1.02, fontsize=17
    # )
    fig.tight_layout()

    indicator = alpha_right * (n_max ** (1.0 - beta))
    print(f"[sanity] alpha_right * n_max^(1-beta) = {indicator:.4g}  (smaller => linear approx more reliable)")
    print("[note] If MC shows NaN: conditioning event too rare -> reduce n_max or increase mc_samples.")

    plt.show()

if __name__ == "__main__":
    plot_hmin_two_panel(
        n_max=10,       # keep moderate so MC conditioning isn't too rare (esp. when r is small)
        beta=0.3,
        sigma_w=1.0,
        sigma_f_list=(0.05, 0.1, 0.3),
        r_left=0.1,
        r_list=(0.0, 0.1, 0.3, 0.5),
        sigma_f_right=0.3,
        b=1,
        mc_samples=10_000_000,
        seed=1334,
    )
