"""Energy measurement on Apple Silicon via `powermetrics` (requires a sudoers rule).

`PowerMeter` samples the combined CPU+GPU+ANE power of the SoC every `interval_ms`
and integrates it over the measured block. The machine-level power is recorded,
so an idle baseline must be measured and subtracted by the caller.

Carbon conversion uses the grid carbon intensity shipped with CodeCarbon.
"""
from __future__ import annotations

import plistlib
import subprocess
import threading
import time

import numpy as np


class PowerMeter:
    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.samples: list[tuple[float, float]] = []   # (elapsed s of sample, combined power W)
        self._proc = None
        self._thread = None

    def _reader(self):
        buf = b""
        for chunk in iter(lambda: self._proc.stdout.read(4096), b""):
            buf += chunk
            while b"\x00" in buf:                          # plists are NUL-separated
                doc, buf = buf.split(b"\x00", 1)
                doc = doc.strip()
                if not doc:
                    continue
                try:
                    p = plistlib.loads(doc)
                except Exception:
                    continue
                proc = p.get("processor", {})
                mw = proc.get("combined_power")
                if mw is None:
                    mw = sum(proc.get(k, 0.0) for k in ("cpu_power", "gpu_power", "ane_power"))
                self.samples.append((p.get("elapsed_ns", self.interval_ms * 1e6) / 1e9, mw / 1000.0))

    def __enter__(self):
        self.samples = []
        self._proc = subprocess.Popen(
            ["sudo", "-n", "/usr/bin/powermetrics", "--samplers", "cpu_power,gpu_power,ane_power",
             "-i", str(self.interval_ms), "-f", "plist"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        time.sleep(2 * self.interval_ms / 1000)            # let the first sample arrive
        self.samples = []
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.duration = time.perf_counter() - self.t0
        time.sleep(self.interval_ms / 1000)                  # capture the last interval
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        self._thread.join(timeout=2)
        p = np.array([w for _, w in self.samples]) if self.samples else np.array([np.nan])
        self.mean_power = float(np.mean(p))                   # W
        self.energy = self.mean_power * self.duration         # J (mean power x wall time)
        self.n_samples = len(self.samples)
        return False


def carbon_intensity(country_iso: str) -> float:
    """Grid carbon intensity in gCO2e/kWh from the CodeCarbon energy-mix data."""
    from codecarbon.input import DataSource
    mix = DataSource().get_global_energy_mix_data()[country_iso]
    return float(mix["carbon_intensity"])
