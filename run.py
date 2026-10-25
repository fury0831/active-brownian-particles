# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Fengrui Zhang

"""Run the ABP parameter scans and reliability studies from a local config.

This entry point runs synthetic-data experiments from an explicit configuration.
The original report-scale experiment has not been rerun for this release.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _on_grid(value, step, name):
    ratio = value / step
    if not math.isclose(ratio, round(ratio), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{name} must be an integer multiple of the time step")


def validate_config(config):
    """Validate the layout supported by the extracted report plotting code."""
    if config["rotation_rates"] != [0.25, 0.5, 1.0, 2.0, 4.0]:
        raise ValueError("This plotting layout requires the original five rotation rates")
    if config["speeds"] != [0.5, 1.0, 2.0]:
        raise ValueError("This plotting layout requires the original three speeds")
    if config["reference_speed"] != 1.0 or config["reference_rotation_rate"] != 1.0:
        raise ValueError("The reference speed and rotation rate must both be 1")
    _integer(config["master_seed"], "master_seed", 0)
    step = config["dimensionless_step"]
    _positive_number(step, "dimensionless_step")
    cc, mc = config["correlation"], config["msd"]
    _integer(cc["trajectories"], "correlation.trajectories", 2)
    _integer(cc["bootstrap_replicates"], "correlation.bootstrap_replicates", 2)
    _integer(mc["trajectories"], "msd.trajectories", 6)
    _integer(mc["batches"], "msd.batches", 2)
    if mc["trajectories"] % mc["batches"]:
        raise ValueError("MSD trajectories must divide into equal batches")
    for name, value in (
        ("correlation.duration_in_tau", cc["duration_in_tau"]),
        ("correlation.maximum_lag_in_tau", cc["maximum_lag_in_tau"]),
        ("msd.duration_in_tau", mc["duration_in_tau"]),
        ("msd.record_interval_in_tau", mc["record_interval_in_tau"]),
    ):
        _positive_number(value, name)
        _on_grid(value, step, name)
    if cc["maximum_lag_in_tau"] > cc["duration_in_tau"]:
        raise ValueError("The maximum correlation lag exceeds the record duration")
    if cc["maximum_lag_in_tau"] < 4:
        raise ValueError("The original correlation-window study requires lags through 4 tau")
    if mc["record_interval_in_tau"] > mc["duration_in_tau"]:
        raise ValueError("The MSD recording interval exceeds the duration")
    for name, section, available, spacing in (
        ("correlation", cc, cc["maximum_lag_in_tau"], step),
        ("msd", mc, mc["duration_in_tau"], mc["record_interval_in_tau"]),
    ):
        window = section["fit_window_in_tau"]
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError(f"{name}.fit_window_in_tau must contain two values")
        low, high = window
        for value in window:
            _positive_number(value, f"{name}.fit_window_in_tau")
            _on_grid(value, spacing, f"{name}.fit_window_in_tau")
        if not 0 < low < high <= available or high - low < 2 * spacing:
            raise ValueError(f"{name} fit window must contain at least three available samples")
    batch_size = mc["trajectories"] // mc["batches"]
    for count in mc["sample_sizes"]:
        _integer(count, "msd.sample_sizes", 2 * batch_size)
        if count > mc["trajectories"] or count % batch_size:
            raise ValueError("Sample-size subsets must contain whole batches within the main run")
    for window in mc["alternative_fit_windows_in_tau"]:
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError("Each alternative MSD fit window must contain two values")
        low, high = window
        for value in window:
            _positive_number(value, "alternative MSD fit window")
            _on_grid(value, mc["record_interval_in_tau"], "alternative MSD fit window")
        if not 0 < low < high <= mc["duration_in_tau"]:
            raise ValueError("An alternative MSD fit window exceeds the recorded duration")
        if high - low < 2 * mc["record_interval_in_tau"]:
            raise ValueError("An alternative MSD fit window has fewer than three samples")
    cv = config["convergence"]
    _integer(cv["trajectories"], "convergence.trajectories", 2)
    _integer(cv["batches"], "convergence.batches", 2)
    if cv["trajectories"] % cv["batches"]:
        raise ValueError("Convergence trajectories must divide into equal batches")
    steps = cv["steps_in_tau"]
    if not isinstance(steps, list) or len(set(steps)) != len(steps):
        raise ValueError("Convergence steps must be a list of distinct values")
    if 0.02 not in steps or 0.01 not in steps:
        raise ValueError("The original refinement summary requires steps 0.02 and 0.01")
    for value in steps:
        _positive_number(value, "convergence.steps_in_tau")
    fine = min(steps)
    for step_value in steps:
        _on_grid(step_value, fine, "nested convergence steps")
        _on_grid(cv["record_interval_in_tau"], step_value, "convergence recording interval")
    for name in ("duration_in_tau", "record_interval_in_tau"):
        _positive_number(cv[name], f"convergence.{name}")
        _on_grid(cv[name], fine, f"convergence.{name}")
    _on_grid(cv["duration_in_tau"], cv["record_interval_in_tau"], "convergence duration")
    low, high = cv["fit_window_in_tau"]
    for value in (low, high):
        _positive_number(value, "convergence fit window")
        _on_grid(value, cv["record_interval_in_tau"], "convergence fit window")
    if not 0 < low < high <= cv["duration_in_tau"]:
        raise ValueError("The convergence fit window exceeds the recorded duration")
    if high - low < 2 * cv["record_interval_in_tau"]:
        raise ValueError("The convergence fit window has fewer than three samples")


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def tau_scaling(rows):
    """Weighted log-log slope, adapted from reproduce_report.ipynb cell 4."""
    import numpy as np

    x = np.array([row["DR"] for row in rows])
    y = np.array([row["tau_fit"] for row in rows])
    se = np.array([row["tau_bootstrap_se"] for row in rows]) / y
    if not np.isfinite(se).all() or np.any(se <= 0):
        raise ValueError("Scaling fit needs positive finite bootstrap standard errors")
    design = np.column_stack([np.log(x), np.ones(len(x))]) / se[:, None]
    coefficients = np.linalg.lstsq(design, np.log(y) / se, rcond=None)[0]
    half = 1.96 * np.sqrt(np.linalg.inv(design.T @ design)[0, 0])
    return [float(coefficients[0]), float(coefficients[0] - half),
            float(coefficients[0] + half)]


def run_scans(config, output, make_plots=True):
    """Generate new results when explicitly invoked by the user."""
    import numpy as np
    import scipy

    from abp import (
        analyze_correlation,
        confidence_from_independent,
        coupled_step_study,
        diffusion_fit,
        exponential_fit,
        mean_and_se,
        simulate_correlation,
        simulate_msd,
    )

    validate_config(config)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Refusing to overwrite a nonempty output location: {output}")
    (output / "data").mkdir(parents=True, exist_ok=True)
    _write_json(output / "config.json", config)
    seeds = [
        int(seed.generate_state(1, dtype=np.uint64)[0])
        for seed in np.random.SeedSequence(config["master_seed"]).spawn(40)
    ]
    summary = {
        "analysis_scope": (
            "Correlation and MSD parameter scans, alternative fitting windows, "
            "sample-size subsets, shared-Brownian-path step refinement, "
            "and the persistence-time scaling fit."
        ),
        "correlation": [],
        "diffusion": [],
    }
    tag = lambda value: f"{value:g}".replace(".", "p")
    cc, mc = config["correlation"], config["msd"]
    step = config["dimensionless_step"]

    for i, rate in enumerate(config["rotation_rates"]):
        data = simulate_correlation(
            rate, step, cc["trajectories"], cc["duration_in_tau"],
            cc["maximum_lag_in_tau"], seeds[2 * i],
        )
        row, mean, se, boot = analyze_correlation(
            data, rate, cc["fit_window_in_tau"],
            cc["bootstrap_replicates"], seeds[2 * i + 1],
        )
        summary["correlation"].append(row)
        data.update(mean=mean, se=se, bootstrap_tau=boot)
        np.savez_compressed(output / "data" / f"correlation_DR_{tag(rate)}.npz", **data)
        if rate == 1.0:
            summary["correlation_windows"] = []
            for low, high in [[0.06, 2], [0.1, 3], [0.2, 4]]:
                mask = (data["time"] >= low - 1e-10) & (data["time"] <= high + 1e-10)
                fitted_tau = exponential_fit(data["time"][mask], mean[mask])
                summary["correlation_windows"].append([low, high, fitted_tau])

    pairs = [(1.0, rate) for rate in config["rotation_rates"]]
    pairs += [(speed, 1.0) for speed in config["speeds"] if speed != 1.0]
    for i, (speed, rate) in enumerate(pairs):
        data = simulate_msd(
            speed, rate, step, mc["trajectories"], mc["batches"],
            mc["duration_in_tau"], mc["record_interval_in_tau"], seeds[10 + i],
        )
        row, batch_d = diffusion_fit(data, mc["fit_window_in_tau"])
        mean, se = mean_and_se(data["batch_msd"])
        data.update(mean=mean, se=se, batch_D=batch_d)
        summary["diffusion"].append(row)
        np.savez_compressed(
            output / "data" / f"msd_v_{tag(speed)}_DR_{tag(rate)}.npz", **data,
        )
        if (speed, rate) == (1.0, 1.0):
            summary["fit_windows"] = [
                diffusion_fit(data, window)[0]
                for window in mc["alternative_fit_windows_in_tau"] + [mc["fit_window_in_tau"]]
            ]
            summary["sample_sizes"] = []
            for count in mc["sample_sizes"]:
                batches = count // (mc["trajectories"] // mc["batches"])
                subset = dict(data, batch_msd=data["batch_msd"][:batches],
                              paths=np.asarray(count))
                summary["sample_sizes"].append(diffusion_fit(subset, mc["fit_window_in_tau"])[0])

    cv = config["convergence"]
    coupled = coupled_step_study(
        cv["steps_in_tau"], cv["trajectories"], cv["batches"],
        cv["duration_in_tau"], cv["record_interval_in_tau"], seeds[17],
    )
    estimates = []
    summary["step_study"] = []
    for i, time_step in enumerate(cv["steps_in_tau"]):
        data = dict(time=coupled["time"], batch_msd=coupled["batch_msd"][i],
                    speed=1.0, DR=1.0, dt=time_step, paths=cv["trajectories"])
        row, batch_d = diffusion_fit(data, cv["fit_window_in_tau"])
        summary["step_study"].append(row)
        estimates.append(batch_d)
    i, j = cv["steps_in_tau"].index(0.02), cv["steps_in_tau"].index(0.01)
    summary["refinement_pct"] = confidence_from_independent(
        (estimates[i] - estimates[j]) / 0.5 * 100
    )
    summary["tau_scaling"] = tau_scaling(summary["correlation"])
    np.savez_compressed(output / "data" / "coupled_step_study.npz",
                        **coupled, batch_D=np.asarray(estimates))

    _write_json(output / "summary.json", summary)
    environment = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    if make_plots:
        import matplotlib
        from figures import make_report_figures

        make_report_figures(output, output / "figures")
        environment["matplotlib"] = matplotlib.__version__
    _write_json(output / "environment.json", environment)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "examples" / "small_config.json",
        help="JSON config; defaults to the newly prepared, reduced-size example",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results" / "demo",
        help="New or empty output directory; nonempty directories are refused",
    )
    parser.add_argument("--no-plots", action="store_true", help="Save numerical results only")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = run_scans(config, args.output, make_plots=not args.no_plots)
    print(f"Saved {len(summary['correlation'])} correlation and "
          f"{len(summary['diffusion'])} MSD parameter sets to {args.output}")
    print(summary["analysis_scope"])


if __name__ == "__main__":
    main()
