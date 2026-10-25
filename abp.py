# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Fengrui Zhang
# Extracted from abp.ipynb code cells; see README.md.

"""Simulation and statistics for the two-dimensional active Brownian particle.

Only the angle is driven by noise. Units are arbitrary but consistent.
The position integrator uses the angle at the LEFT endpoint of each step.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.optimize import least_squares
from scipy.stats import t as student_t

def correlation_theory(time, rotation_rate):
    return np.exp(-rotation_rate * np.abs(np.asarray(time)))


def msd_theory(time, speed, rotation_rate):
    """Exact continuous-time MSD, including the ballistic/diffusive crossover."""
    time = np.asarray(time, dtype=float)
    if rotation_rate == 0:
        return (speed * time) ** 2
    u = rotation_rate * time
    # Avoid cancellation between u and expm1(-u) for very small u.
    f = np.where(
        np.abs(u) < 1e-4,
        u**2 / 2 - u**3 / 6 + u**4 / 24 - u**5 / 120,
        u + np.expm1(-u),
    )
    return 2 * speed**2 / rotation_rate**2 * f


def discrete_diffusion(speed, rotation_rate, step):
    """Long-time diffusion coefficient of the left-endpoint integrator."""
    if rotation_rate <= 0 or step <= 0:
        raise ValueError("A positive rotation rate and time step are required.")
    return speed**2 * step / (4 * np.tanh(rotation_rate * step / 2))


def grid_count(duration, step):
    count = int(round(duration / step))
    if count <= 0 or not np.isclose(count * step, duration):
        raise ValueError("Duration must be a positive integer multiple of the step.")
    return count

def directional_autocorrelation(phi, max_lag):
    """Unbiased, non-circular time autocorrelation, one row per trajectory.

    Do NOT subtract each trajectory's empirical direction mean: this observable
    is <n(s+t).n(s)>, not a covariance of demeaned finite records.
    Zero padding prevents FFT wrap-around. Divide lag k by L-k.
    """
    phi = np.atleast_2d(np.asarray(phi, dtype=float))
    length = phi.shape[1]
    if not 0 <= max_lag < length:
        raise ValueError("Lag must be smaller than the number of samples.")
    fft_size = next_fast_len(2 * length - 1)
    out = np.zeros((phi.shape[0], max_lag + 1))
    for component in (np.cos(phi), np.sin(phi)):
        transform = rfft(component, n=fft_size, axis=1)
        out += irfft(transform * transform.conj(), n=fft_size, axis=1)[
            :, : max_lag + 1
        ]
    return out / (length - np.arange(max_lag + 1))


def simulate_correlation(rotation_rate, h, paths, duration_u, max_lag_u, seed):
    """Simulate independent stationary angular records and time-average each."""
    rng = np.random.default_rng(seed)
    steps = grid_count(duration_u, h)
    max_lag = grid_count(max_lag_u, h)
    curves = []
    # Small chunks cap FFT memory. Each chunk contains independent trajectories.
    for start in range(0, paths, 16):
        count = min(16, paths - start)
        increments = rng.normal(0, np.sqrt(2 * h), size=(count, steps))
        phi = np.empty((count, steps + 1))
        phi[:, 0] = rng.uniform(0, 2 * np.pi, count)
        phi[:, 1:] = phi[:, :1] + np.cumsum(increments, axis=1)
        curves.append(directional_autocorrelation(phi, max_lag))
    return {
        "time": np.arange(max_lag + 1) * h / rotation_rate,
        "per_path": np.concatenate(curves),
    }

def mean_and_se(curves):
    curves = np.asarray(curves)
    if curves.shape[0] < 2:
        raise ValueError("At least two independent curves are required.")
    return curves.mean(axis=0), curves.std(axis=0, ddof=1) / np.sqrt(len(curves))


def confidence_from_independent(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    half = float(student_t.ppf(0.975, len(values) - 1) * se)
    return mean, se, mean - half, mean + half


def exponential_fit(time, corr):
    """Fit exp(-t/tau) in data space, with C(0)=1 and tau strictly positive.

    Start from the observed 1/e crossing, without fixing tau to its theory value.
    Optimizing log(tau) enforces positivity and avoids a logarithm of noisy data.
    """
    time, corr = np.asarray(time), np.asarray(corr)
    initial = time[np.argmin(np.abs(corr - np.exp(-1)))]
    initial = max(float(initial), float(time[0]), 1e-12)

    def residual(z):
        return np.exp(-time / np.exp(z[0])) - corr

    fit = least_squares(residual, [np.log(initial)], xtol=1e-10, ftol=1e-10)
    if not fit.success:
        raise RuntimeError(f"Exponential fit failed: {fit.message}")
    return float(np.exp(fit.x[0]))

def analyze_correlation(data, rotation_rate, window_u, bootstrap_replicates, seed):
    time, per_path = data["time"], data["per_path"]
    mean, se = mean_and_se(per_path)
    u = time * rotation_rate
    mask = (u >= window_u[0] - 1e-10) & (u <= window_u[1] + 1e-10)
    tau = exponential_fit(time[mask], mean[mask])
    rng = np.random.default_rng(seed)
    # Resample WHOLE independent trajectories, retaining all lag correlations.
    fit_curves = per_path[:, mask]
    boot = np.array([
        exponential_fit(time[mask], fit_curves[rng.integers(
            0, len(per_path), len(per_path)
        )].mean(axis=0))
        for _ in range(bootstrap_replicates)
    ])
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "DR": float(rotation_rate),
        "tau_theory": 1 / rotation_rate,
        "tau_fit": tau,
        "tau_ci_low": float(low),
        "tau_ci_high": float(high),
        "tau_bootstrap_se": float(boot.std(ddof=1)),
        "tau_relative_error_pct": 100 * (tau * rotation_rate - 1),
        "fit_rmse": float(np.sqrt(np.mean(
            (mean[mask] - np.exp(-time[mask] / tau)) ** 2
        ))),
    }, mean, se, boot

def record_indices(steps, h, record_u):
    stride = grid_count(record_u, h)
    # Keep every early step to resolve the ballistic regime; thin the tail.
    return np.unique(np.r_[
        np.arange(min(steps, round(1 / h)) + 1),
        np.arange(0, steps + 1, stride),
        steps,
    ]).astype(int)


def simulate_msd(speed, rotation_rate, h, paths, batches, duration_u, record_u, seed):
    """Stream independent trajectories, storing batch MSDs and six sample paths."""
    if paths % batches or rotation_rate <= 0:
        raise ValueError("Paths must divide into batches; DR must be positive.")
    rng = np.random.default_rng(seed)
    steps = grid_count(duration_u, h)
    record = record_indices(steps, h, record_u)
    dt = h / rotation_rate
    phi = rng.uniform(0, 2 * np.pi, paths)
    x, y = np.zeros(paths), np.zeros(paths)
    batch_msd = np.zeros((batches, len(record)))
    sample_xy = np.zeros((6, len(record), 2))
    slot = 1
    for step in range(1, steps + 1):
        x += speed * dt * np.cos(phi)
        y += speed * dt * np.sin(phi)
        phi += np.sqrt(2 * h) * rng.standard_normal(paths)
        if step == record[slot]:
            batch_msd[:, slot] = (x*x + y*y).reshape(batches, -1).mean(axis=1)
            sample_xy[:, slot, 0] = x[:6]
            sample_xy[:, slot, 1] = y[:6]
            slot += 1
            if slot == len(record):
                break
    return {
        "time": record * dt,
        "batch_msd": batch_msd,
        "sample_xy": sample_xy,
        "speed": np.asarray(speed),
        "DR": np.asarray(rotation_rate),
        "dt": np.asarray(dt),
        "paths": np.asarray(paths),
    }

def diffusion_fit(data, window_u):
    """Fit each independent batch, then infer uncertainty across batches."""
    time = data["time"]
    u = time * float(data["DR"])
    mask = (u >= window_u[0] - 1e-10) & (u <= window_u[1] + 1e-10)
    if mask.sum() < 3:
        raise ValueError("At least three time samples are needed for a line fit.")
    design = np.column_stack([time[mask], np.ones(mask.sum())])
    coefficients = np.linalg.lstsq(
        design, data["batch_msd"][:, mask].T, rcond=None
    )[0]
    estimates = coefficients[0] / 4
    mean, se, low, high = confidence_from_independent(estimates)
    v, dr = float(data["speed"]), float(data["DR"])
    theory = v*v / (2 * dr)
    return {
        "v": v, "DR": dr, "dt": float(data["dt"]),
        "paths": int(data["paths"]),
        "fit_start_in_tau": float(window_u[0]),
        "fit_end_in_tau": float(window_u[1]),
        "D_theory": theory, "D_fit": mean, "D_se": se,
        "D_ci_low": low, "D_ci_high": high,
        "D_relative_error_pct": 100 * (mean / theory - 1),
        "intercept": float(coefficients[1].mean()),
    }, estimates

def coupled_step_study(h_values, paths, batches, duration_u, record_u, seed):
    """Nested Brownian increments couple all time steps to the same paths.

    Parameters here are v=DR=1. Coarse/fine differences thus isolate integration
    error far more efficiently than unrelated random seeds would.
    """
    h_values = np.asarray(h_values)
    fine = float(min(h_values))
    ratios = np.rint(h_values / fine).astype(int)
    if paths % batches or not np.allclose(ratios * fine, h_values):
        raise ValueError("Steps must be nested and batches must have equal size.")
    steps = grid_count(duration_u, fine)
    record_stride = grid_count(record_u, fine)
    if np.any(record_stride % ratios):
        raise ValueError("Recording times must coincide with every coarse grid.")
    time = np.arange(0, steps + 1, record_stride) * fine
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, paths)
    x = np.zeros((len(h_values), paths))
    y = np.zeros_like(x)
    curves = np.zeros((len(h_values), batches, len(time)))
    for step in range(steps):
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        for level, (h, ratio) in enumerate(zip(h_values, ratios)):
            if step % ratio == 0:
                x[level] += h * cos_phi
                y[level] += h * sin_phi
        phi += np.sqrt(2 * fine) * rng.standard_normal(paths)
        if (step + 1) % record_stride == 0:
            curves[:, :, (step + 1) // record_stride] = (
                x*x + y*y
            ).reshape(len(h_values), batches, -1).mean(axis=2)
    return {"time": time, "batch_msd": curves, "h_values": h_values}
