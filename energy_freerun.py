"""Energy, carbon and time of the free-run simulation of the identified models.

Separate from the complexity study: it only READS ``Resultados/<cost>/runs.json`` (the runs of
``MGGP_multiobjetivo.ipynb`` for each cost scenario) and never reruns the evolutionary search. For
every run the model chosen by the tau-rule is rebuilt, re-estimated on the identification data and
simulated in free run over the validation input, exactly as for the validation NRMSE of the paper.

Measured configurations (``ECONFIGS``): SO (independent of the cost), MO-lex and Green with the
operation-count cost (rho = 1, primary) and MO-lex and Green with the Karatsuba weighting
(rho = 729/64), labelled ``MO-lex-K`` and ``Green-K``.

Two simulators are measured:

* ``canonical`` -- straight-line Python generated from the canonical, duplicate-free polynomial,
                   executing exactly a = p-1 additions and b = sum_i d_i multiplications per
                   sample, i.e. the operations counted by the cost (primary);
* ``package``   -- the package predictor ``predict("FreeRun", ...)``; it evaluates every gene,
                   duplicated genes included, with NumPy windows (what the package user executes).

Energy is measured in blocks: one block repeats the free-run simulation of the ~30 selected models
of one (example, configuration, simulator) group, one full cycle at a time, for at least
``block_s`` seconds, while ``power_meter.PowerMeter`` (powermetrics, 100 ms) records the machine.
Idle blocks (sleep) bracket every round, and the **median idle power of all idle blocks** is
subtracted (robust to a block disturbed by background activity); the round-wise bracket mean is
kept as a sensitivity. Rounds visit the groups in a seeded random order. CodeCarbon supplies only
the grid carbon intensity (``power_meter.carbon_intensity``).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (ROOT, os.path.join(ROOT, "mggp_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("TQDM_DISABLE", "1")

from mggp import MGGP                                             # noqa: E402
from utils.utils import simulate                                  # noqa: E402  (Bouc-Wen of the package)
from green_mggp import COST_SCENARIOS, gene_monomial, make_cost  # noqa: E402
from power_meter import PowerMeter, carbon_intensity              # noqa: E402

RES = os.path.join(ROOT, "Resultados")
OUT = os.environ.get("GMGGP_ENERGY_OUT", os.path.join(RES, "energia"))
CONFIGS = ["SO", "MO-lex", "Green"]
# energy configurations: (label, configuration in the study, cost scenario)
ECONFIGS = [("SO", "SO", "ops"), ("MO-lex", "MO-lex", "ops"), ("Green", "Green", "ops"),
            ("MO-lex-K", "MO-lex", "karatsuba"), ("Green-K", "Green", "karatsuba")]
ELABELS = [c[0] for c in ECONFIGS]
RHO_K = COST_SCENARIOS["karatsuba"]
SIMULATORS = ["package", "canonical"]
TAU, ETA = 0.05, 1e-6                                             # declared tau-rule (unchanged)


# ---------------------------------------------------------------------------
# Data and settings: verbatim copies of MGGP_multiobjetivo.ipynb (section 2)
# ---------------------------------------------------------------------------
def tutorial_data(noise_std=0.0, noise_seed=2027):
    n = 300
    t = np.linspace(0, 20, n)
    u = (np.sin(t) + 0.25 * np.sin(3 * t)).reshape(-1, 1)
    y = np.zeros((n, 1))
    for k in range(1, n):
        y[k, 0] = 0.65 * y[k - 1, 0] + 0.35 * u[k, 0] + 0.10 * u[k, 0] * y[k - 1, 0]
    y = y + noise_std * np.random.default_rng(noise_seed).standard_normal(y.shape)
    s = int(0.70 * n)
    return dict(u_id=u[:s], y_id=y[:s], u_val=u[s:], y_val=y[s:])


def boucwen_data(D=20, h=1e-3):
    def sim(T, f, amp):
        t = np.arange(0, T, h)
        u = amp(t) * np.sin(2 * np.pi * f * t)
        y = simulate(u, np.gradient(u, h))
        return u[::D].reshape(-1, 1), y[::D].reshape(-1, 1)
    u_id, y_id = sim(6.0, 1.0, lambda t: 20 + 5 * t)
    u_val, y_val = sim(4.0, 0.8, lambda t: 45 + 0 * t)
    return dict(u_id=u_id, y_id=y_id, u_val=u_val, y_val=y_val)


COMMON = dict(problem_type="regression", mode="NARX", evaluationMode="RMSE",
              evaluationType="MShooting", evaluationTypeTest="FreeRun", k=20,
              generations=50, populationSize=100, mutationRate=0.2, crossoverRate=0.9,
              elitePercentage=10, operators=["mul"])


def examples():
    return {
        "E1": dict(data=tutorial_data(0.0), search=dict(nTerms=5, maxHeight=2, nDelays=2)),
        "E2": dict(data=tutorial_data(0.02), search=dict(nTerms=5, maxHeight=2, nDelays=2)),
        "E3": dict(data=boucwen_data(), search=dict(nTerms=8, maxHeight=3, nDelays=3)),
    }


def select_tau(models, tau=TAU):
    jmin = min(m["J"] for m in models)
    ok = [m for m in models if m["J"] <= (1 + tau) * jmin + ETA]
    return min(ok, key=lambda m: (m["C"], m["J"]))


def nrmse(yd, yp):
    yd, yp = np.ravel(yd), np.ravel(yp)
    if not np.all(np.isfinite(yp)):
        return np.inf
    return float(np.sqrt(np.mean((yd - yp) ** 2)) / np.std(yd))


# ---------------------------------------------------------------------------
# Rebuilding the selected models and the two simulators
# ---------------------------------------------------------------------------
def rebuild(genes, ex):
    """Package model with coefficients re-estimated on the identification data (as in the notebook)."""
    d = ex["data"]
    cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp())                                  # the package writes models_saved/ in cwd
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            mg = MGGP(inputs=d["u_id"], outputs=d["y_id"],
                      **{**COMMON, "generations": 1, "populationSize": 2}, **ex["search"])
            model = mg.element.buildModelFromList(list(genes))
            mg.element.compileModel(model)
            model.theta = list(model.leastSquares(d["y_id"], d["u_id"]))
    finally:
        os.chdir(cwd)
    return model


def canonical_source(model):
    """Python source of the canonical polynomial: duplicated genes merged into one coefficient.

    Each term is written  c*f1*...*fd  (d multiplications) and the terms are chained by p-1
    additions after the constant, so one sample executes exactly the operations counted by C.
    """
    theta = [float(np.ravel(t)[0]) for t in model.theta]
    coef = OrderedDict()
    for tree, th in zip(model, theta[1:]):
        mono = gene_monomial(tree)
        coef[mono] = coef.get(mono, 0.0) + th
    need = 0
    terms = []
    for mono, c in coef.items():
        factors = []
        for (var, shift), pw in mono:
            if var.startswith("y"):
                factors += [f"y[k-{shift + 1}]"] * pw
                need = max(need, shift + 1)
            else:
                factors += [f"u[k-{shift}]" if shift else "u[k]"] * pw
                need = max(need, shift)
        terms.append("*".join([repr(c)] + factors))
    # The package free run keeps only lagMax past outputs and delays them with np.roll, so a y factor
    # whose delay equals lagMax wraps around (y(k-1-lagMax) is read as y(k-1)). The canonical
    # simulator uses the intended delays and therefore may need one more initial sample.
    lag = max(int(model.lagMax), need)
    expr = " + ".join([repr(theta[0])] + terms)
    src = (f"def _sim(y0, u, n):\n"
           f"    y = list(y0)\n"
           f"    for k in range({lag}, n):\n"
           f"        y.append({expr})\n"
           f"    return y\n")
    return src, lag, len(coef), sum(sum(p for _, p in m) for m in coef)


def make_simulators(model, ex):
    """Return (package_fn, canonical_fn, info). Both simulate the validation data in free run."""
    d = ex["data"]
    y_val, u_val = d["y_val"], d["u_val"]
    src, lag, p1, sumd = canonical_source(model)
    ns = {}
    exec(src, ns)
    sim = ns["_sim"]
    u_list = [float(v) for v in np.ravel(u_val)]
    y0 = [float(v) for v in np.ravel(y_val)[:lag]]
    n = len(u_list)

    def package_fn():
        return model.predict("FreeRun", y_val, u_val)

    def canonical_fn():
        return sim(y0, u_list, n)

    yp_pkg, yd = package_fn()
    yp_pkg = np.ravel(yp_pkg)
    yp_can = np.array(canonical_fn()[lag:])
    skip = lag - int(model.lagMax)                                    # align both outputs at sample `lag`
    info = dict(lag=lag, lag_package=int(model.lagMax), p1_canon=p1, sumd_canon=sumd, n_steps=n - lag,
                n_genes=len(model), val_package=nrmse(yd, yp_pkg),
                val_canonical=nrmse(np.ravel(yd)[skip:], yp_can),
                max_abs_diff=float(np.max(np.abs(yp_pkg[skip:] - yp_can))) if len(yp_can) else np.nan,
                source=src)
    return package_fn, canonical_fn, info


def load_models(res_dir=RES, econfigs=ECONFIGS):
    """One record per (energy configuration, run): the tau-selected model, rebuilt, with both simulators.

    ``C`` is the cost of the model in its own scenario; ``C_ops = a + b`` and ``C_K = a + rho_K b``
    are the costs under both weights. Models identical across configurations share the simulators.
    """
    exs = examples()
    runs = {}
    cache, rows = {}, []
    for label, config, cost in econfigs:
        if cost not in runs:
            runs[cost] = json.load(open(os.path.join(res_dir, cost, "runs.json")))
        cost_fn = make_cost(COST_SCENARIOS[cost])
        for r in runs[cost]:
            if r["config"] != config:
                continue
            ex = exs[r["example"]]
            sel = select_tau(r["delivered"])
            row = dict(example=r["example"], config=label, seed=r["seed"], cost=cost, C=sel["C"], p1=sel["p1"],
                       sumd=sel["sumd"], C_ops=sel["p1"] + sel["sumd"], C_K=sel["p1"] + RHO_K * sel["sumd"],
                       val_saved=sel["val"], genes=tuple(sel["genes"]), terms=tuple(sel["terms"]))
            if not np.isfinite(sel["val"]):                          # failed in the study: kept as missing
                row.update(ok=False, reason="non-finite validation NRMSE in runs.json")
                rows.append(row)
                continue
            key = (r["example"], row["genes"])
            if key not in cache:
                model = rebuild(row["genes"], ex)
                cache[key] = (model, *make_simulators(model, ex))
            model, pkg, can, info = cache[key]
            row.update(ok=True, C_rebuilt=float(cost_fn(model)), pkg=pkg, can=can,
                       **{k: v for k, v in info.items() if k != "source"})
            rows.append(row)
    return rows


def check_models(rows, tol=1e-9):
    """Rebuilt models must reproduce the saved cost and validation NRMSE; the two simulators must agree."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("pkg", "can", "terms")} for r in rows])
    ok = df[df.ok]
    report = dict(
        n_records=len(df), n_measured=len(ok), n_missing=int((~df.ok).sum()),
        missing=df[~df.ok][["config", "example", "seed"]].astype(str).agg(" ".join, axis=1).tolist(),
        cost_mismatch=int((np.abs(ok.C - ok.C_rebuilt) > 1e-9).sum()),
        val_mismatch=int((np.abs(ok.val_saved - ok.val_package) > tol).sum()),
        max_val_diff=float(np.max(np.abs(ok.val_saved - ok.val_package))),
        max_sim_diff=float(ok.max_abs_diff.max()),
        n_unique_package=int(ok[["example", "genes"]].drop_duplicates().shape[0]),
    )
    return df, report


# ---------------------------------------------------------------------------
# Timing per model (no power measurement needed)
# ---------------------------------------------------------------------------
def time_per_sim(fn, min_time=0.05, repeat=5):
    """Median wall time of one call over `repeat` batches, each lasting at least `min_time` s."""
    fn()
    n, t = 1, 0.0
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        t = time.perf_counter() - t0
        if t >= min_time:
            break
        n *= 2
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        ts.append((time.perf_counter() - t0) / n)
    return float(np.median(ts))


def timing_table(rows, min_time=0.05, repeat=5):
    out, cache = [], {}
    for r in rows:
        base = {k: r[k] for k in ("example", "config", "seed", "cost", "C", "C_ops", "C_K", "p1", "sumd")}
        if not r["ok"]:
            out.append({**base, "t_package": np.nan, "t_canonical": np.nan})
            continue
        key = (r["example"], r["genes"])
        if key not in cache:
            cache[key] = (time_per_sim(r["pkg"], min_time, repeat), time_per_sim(r["can"], min_time, repeat))
        out.append({**base, "n_genes": r["n_genes"], "n_steps": r["n_steps"],
                    "t_package": cache[key][0], "t_canonical": cache[key][1]})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Energy blocks
# ---------------------------------------------------------------------------
def measure_block(fns, block_s, interval_ms=100):
    """Run complete cycles over `fns` for at least `block_s` s (fns=None: idle sleep).

    Returns time, number of simulations and the energy seen by powermetrics.
    """
    with PowerMeter(interval_ms) as pm:
        t0 = time.perf_counter()
        n = 0
        if fns is None:
            time.sleep(block_s)
        else:
            while time.perf_counter() - t0 < block_s:
                for f in fns:
                    f()
                n += len(fns)
        work_s = time.perf_counter() - t0
    return dict(n_sims=n, work_s=work_s, pm_s=pm.duration, pm_J=pm.energy, pm_W=pm.mean_power,
                pm_samples=pm.n_samples)


def run_energy(rows, rounds=10, block_s=10.0, seed=2027, out_file=None, verbose=True):
    """Interleaved energy measurement of all (example, config, simulator) groups.

    Order per round: idle block, then the groups in a seeded random order; a final idle block
    closes the last round. Every block is appended to `out_file` as soon as it is measured.
    """
    out_file = out_file or os.path.join(OUT, "energy_blocks.csv")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    groups = OrderedDict()
    for ex in ["E1", "E2", "E3"]:
        for cfg in ELABELS:
            ok = [r for r in rows if r["example"] == ex and r["config"] == cfg and r["ok"]]
            for sim in SIMULATORS:
                groups[(ex, cfg, sim)] = dict(fns=[r["pkg" if sim == "package" else "can"] for r in ok],
                                              n_models=len(ok))
    rng = random.Random(seed)
    blocks = []

    def record(rnd, pos, ex, cfg, sim, n_models, res):
        blocks.append(dict(round=rnd, position=pos, example=ex, config=cfg, simulator=sim,
                           n_models=n_models, **res))
        pd.DataFrame(blocks).to_csv(out_file, index=False)
        if verbose:
            tag = "idle" if sim == "idle" else f"{ex} {cfg:8s} {sim:9s}"
            print(f"round {rnd} [{pos:2d}] {tag:26s} {res['pm_W']:6.2f} W  n={res['n_sims']}", flush=True)

    t_start = time.time()
    for rnd in range(rounds):
        record(rnd, 0, "", "", "idle", 0, measure_block(None, block_s))
        order = list(groups)
        rng.shuffle(order)
        for pos, key in enumerate(order, start=1):
            g = groups[key]
            record(rnd, pos, *key, g["n_models"], measure_block(g["fns"], block_s))
        if verbose:
            print(f"round {rnd} done ({time.time() - t_start:.0f} s)", flush=True)
    record(rounds, 0, "", "", "idle", 0, measure_block(None, block_s))
    return pd.DataFrame(blocks)


def net_per_sim(blocks, country="IRL"):
    """Idle-subtracted energy, carbon and time per simulation for every measured block.

    Energy is measured by powermetrics (``pm_*``). The idle power subtracted is the **median of all
    idle blocks** of the session, a property of the machine that is robust to a single idle block
    disturbed by background activity. The round-wise alternative (mean of the two idle blocks that
    bracket the round, the original protocol) is kept as a sensitivity column (``*_bracket``).
    CodeCarbon is used only for the grid carbon intensity of `country`.
    """
    b = blocks.copy()
    idle = b[b.simulator == "idle"].set_index("round")
    p_idle = idle.pm_J / idle.pm_s
    work = b[b.simulator != "idle"].copy()
    work["idle_W"] = float(np.median(p_idle))
    work["idle_bracket_W"] = work["round"].map(lambda r: 0.5 * (p_idle[r] + p_idle[r + 1]))
    ci = carbon_intensity(country)                                     # gCO2e/kWh (CodeCarbon data)
    work["t_us"] = 1e6 * work.work_s / work.n_sims
    work["net_W"] = work.pm_J / work.pm_s - work.idle_W
    work["E_uJ"] = 1e6 * (work.pm_J - work.idle_W * work.pm_s) / work.n_sims
    work["E_bracket_uJ"] = 1e6 * (work.pm_J - work.idle_bracket_W * work.pm_s) / work.n_sims
    work["E_gross_uJ"] = 1e6 * work.pm_J / work.n_sims
    work["CO2_ug"] = work.E_uJ / 3.6e6 * ci                            # micro-g CO2e per simulation
    work["carbon_intensity_g_per_kWh"] = ci
    work["country"] = country
    return work


def paired_ratios(per_sim, value="E_uJ"):
    """Per-round ratio of each configuration to SO (same round, example and simulator)."""
    piv = per_sim.pivot_table(index=["example", "simulator", "round"], columns="config", values=value)
    out = []
    for cfg in ELABELS:
        r = (piv[cfg] / piv["SO"]).rename("ratio").reset_index()
        r["config"] = cfg
        out.append(r)
    return pd.concat(out, ignore_index=True)


def ops_regression(df, time_col):
    """Least squares  t = t0 + t_add * a + t_mul * b  (a additions, b multiplications per sample),
    and the single-regressor fits  t = c0 + c1 * C  for C = a + b (rho = 1) and C = a + rho_K b,
    on distinct canonical structures. t_mul / t_add is the calibrated multiplication weight."""
    y = df[time_col].to_numpy()

    def fit(X):
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        return coef, 1 - np.sum((y - X @ coef) ** 2) / np.sum((y - y.mean()) ** 2)
    one = np.ones(len(df))
    coef, r2 = fit(np.c_[one, df.p1, df.sumd])
    c1, r2_1 = fit(np.c_[one, df.p1 + df.sumd])
    cK, r2_K = fit(np.c_[one, df.p1 + RHO_K * df.sumd])
    return dict(t0=coef[0], t_add=coef[1], t_mul=coef[2], mul_over_add=coef[2] / coef[1], R2_ops=r2,
                c0_one=c1[0], c1_one=c1[1], R2_one=r2_1, c0_K=cK[0], c1_K=cK[1], R2_K=r2_K, n=len(df))


def machine_info(country):
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            return ""
    import codecarbon
    return dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), platform=platform.platform(),
                chip=sh(["sysctl", "-n", "machdep.cpu.brand_string"]), model=sh(["sysctl", "-n", "hw.model"]),
                python=sys.version.split()[0], numpy=np.__version__, codecarbon=codecarbon.__version__,
                power_source=sh(["pmset", "-g", "batt"]).splitlines()[0] if sh(["pmset", "-g", "batt"]) else "",
                country=country, carbon_intensity_g_per_kWh=carbon_intensity(country))
