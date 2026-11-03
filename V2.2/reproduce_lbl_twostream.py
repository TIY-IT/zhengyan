"""Reproduce the teacher's multi-window LBL + two-stream experiment.

The default workflow reads HITEMP files from the configured HITEMP root,
supplements NH3 with the local HITRAN files (both isotopologues),
selects the three independent windows recommended for the high-temperature
case, and writes one forward-model/ML dataset under each window directory.
Local labeled LBL files and one legacy mixed-species .par file remain
supported through --data-source.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


import numpy as np
from scipy.special import wofz
from scipy.stats import qmc

from isotopologue_data import ISOTOPOLOGUE_INFO, ISOTOPOLOGUE_LOG_Q_CHEBYSHEV


# SI constants, except C2 which is convenient for HITRAN wavenumbers in cm^-1.
H = 6.62607015e-34
C = 299792458.0
K_B = 1.380649e-23
AMU = 1.66053906660e-27
C2_CM_K = 1.438776877
T_REF = 296.0
P_REF_PA = 101325.0

SPECIES_BY_ID = {1: "H2O", 5: "CO", 6: "CH4", 11: "NH3"}
MOLECULE_ID_BY_SPECIES = {name: molecule_id for molecule_id, name in SPECIES_BY_ID.items()}
MOLECULAR_MASS_AMU = {"H2O": 18.010565, "CO": 27.994915, "CH4": 16.031300, "NH3": 17.026549}

# Partition sums: every isotopologue has its own log Q(T) Chebyshev fit in
# isotopologue_data.py (generated from the HITRAN TIPS-2021 tables by
# tools/fit_isotopologue_tips.py).  z = 2 * (T - 296) / (1800 - 296) - 1.
TIPS_T_MIN = 296.0
TIPS_T_MAX = 1800.0


class ProgressBar:
    """Small dependency-free terminal progress bar for long HITEMP runs."""

    def __init__(self, label: str, total: int, unit: str = "items") -> None:
        self.label = label
        self.total = max(int(total), 1)
        self.unit = unit
        self.started = time.perf_counter()
        self.last_update = 0.0
        self.current = 0

    def update(self, current: int, suffix: str = "", force: bool = False) -> None:
        current = min(max(int(current), 0), self.total)
        self.current = current
        now = time.perf_counter()
        if not force and current < self.total and now - self.last_update < 0.2:
            return
        self.last_update = now
        fraction = current / self.total
        width = 28
        filled = int(width * fraction)
        elapsed = max(now - self.started, 1.0e-9)
        rate = current / elapsed
        remaining = (self.total - current) / rate if rate > 0.0 else float("inf")
        eta = "done" if current >= self.total else f"ETA {remaining:.0f}s"
        bar = "#" * filled + "." * (width - filled)
        detail = f"{current}/{self.total} {self.unit}"
        if suffix:
            detail += f" | {suffix}"
        if os.environ.get("ZY_PROGRESS"):
            # Machine-readable line consumed by dashboard.py (no \r tricks).
            print(f"@@PROGRESS\t{self.label}\t{fraction:.6f}\t{detail}\t{eta}", flush=True)
        else:
            print(
                f"\r{self.label} [{bar}] {fraction * 100:6.2f}% {detail} | {eta}",
                end="",
                flush=True,
            )
        if current >= self.total and not os.environ.get("ZY_PROGRESS"):
            print(flush=True)

    def finish(self, suffix: str = "") -> None:
        if self.current >= self.total:
            return
        self.update(self.total, suffix=suffix, force=True)


@dataclass(frozen=True)
class SpectralWindow:
    """One independent spectral window used by the surrogate workflow."""

    start_cm: float
    stop_cm: float

    @property
    def name(self) -> str:
        return f"window_{self.start_cm:g}_{self.stop_cm:g}"

    @property
    def label(self) -> str:
        return f"{self.start_cm:g}-{self.stop_cm:g} cm-1"


def validate_physical_constants() -> None:
    """Fail early with a useful message if an SI constant is mistyped."""
    expected_ranges = {
        "H": (6.62e-34, 6.64e-34, H),
        "C": (2.99e8, 3.01e8, C),
        "K_B": (1.37e-23, 1.39e-23, K_B),
        "AMU": (1.65e-27, 1.67e-27, AMU),
    }
    invalid = [
        f"{name}={value:.8e}"
        for name, (lower, upper, value) in expected_ranges.items()
        if not lower <= value <= upper
    ]
    if invalid:
        raise ValueError("Invalid physical constant: " + ", ".join(invalid))


@dataclass(frozen=True)
class HitranLines:
    species: str
    molecule_id: int
    isotopologue: np.ndarray
    nu0: np.ndarray
    strength_ref: np.ndarray
    einstein_a: np.ndarray
    gamma_air: np.ndarray
    gamma_self: np.ndarray
    e_lower: np.ndarray
    n_air: np.ndarray
    delta_air: np.ndarray

    def __len__(self) -> int:
        return len(self.nu0)


@dataclass
class TorchGpuWorkspace:
    """CUDA tensors that remain resident across many forward-model samples."""

    torch: Any
    device: Any
    dtype: Any
    dtype_name: str
    device_name: str
    torch_version: str
    nu_grid: Any
    nu_grid_relative: Any
    p_mid: Any
    p_edges: Any
    line_lists: dict[str, dict[str, Any]]
    weideman_l: float
    weideman_coefficients: Any


def _isotopologue_id(char: str) -> int:
    """HITRAN local isotopologue code: 1-9, then 0 = 10, A = 11, B = 12, ..."""
    if char.isdigit():
        return 10 if char == "0" else int(char)
    if char.isalpha():
        return ord(char.upper()) - ord("A") + 11
    raise ValueError(f"无法识别的同位素编号字符 {char!r}")


def _as_float(text: str, default: float = 0.0) -> float:
    text = text.strip()
    return float(text) if text else default


def parse_window_specification(text: str) -> list[SpectralWindow]:
    """Parse ``start-stop,start-stop`` spectral-window specifications."""
    windows: list[SpectralWindow] = []
    for raw_item in text.split(","):
        item = raw_item.strip().replace(":", "-")
        if not item:
            continue
        parts = item.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"波数窗口格式无效：{raw_item!r}；应为 start-stop，例如 1200-1400"
            )
        try:
            start_cm, stop_cm = (float(part.strip()) for part in parts)
        except ValueError as exc:
            raise ValueError(f"波数窗口格式无效：{raw_item!r}") from exc
        if not np.isfinite(start_cm) or not np.isfinite(stop_cm) or stop_cm <= start_cm:
            raise ValueError(f"波数窗口必须满足 stop > start：{raw_item!r}")
        windows.append(SpectralWindow(start_cm, stop_cm))
    if not windows:
        raise ValueError("至少需要一个波数窗口")
    names = [window.name for window in windows]
    if len(set(names)) != len(names):
        raise ValueError("波数窗口不能重复")
    return windows


def _parse_labeled_range(text: str) -> tuple[float, float] | None:
    """Extract ranges such as ``1.2k-1.4k`` or ``1200-1400`` from a path."""
    match = re.search(
        r"(?<!\d)(\d+(?:\.\d+)?)(k?)-(\d+(?:\.\d+)?)(k?)(?!\d)",
        text.lower(),
    )
    if match is None:
        return None
    start = float(match.group(1)) * (1000.0 if match.group(2) else 1.0)
    stop = float(match.group(3)) * (1000.0 if match.group(4) else 1.0)
    return (start, stop) if stop > start else None


def _overlap(left: tuple[float, float], right: SpectralWindow) -> float:
    return max(0.0, min(left[1], right.stop_cm) - max(left[0], right.start_cm))


def _empty_line_records() -> dict[str, list[float]]:
    return {
        "molecule_id": [],
        "isotopologue": [],
        "nu0": [],
        "strength_ref": [],
        "einstein_a": [],
        "gamma_air": [],
        "gamma_self": [],
        "e_lower": [],
        "n_air": [],
        "delta_air": [],
    }


def _records_to_lines(
    records_by_window: dict[str, dict[str, dict[str, list[float]]]],
    windows: list[SpectralWindow],
) -> dict[str, dict[str, HitranLines]]:
    parsed: dict[str, dict[str, HitranLines]] = {}
    for window in windows:
        window_records = records_by_window[window.name]
        parsed_window: dict[str, HitranLines] = {}
        for species, row in window_records.items():
            if not row["nu0"]:
                # Some supplied LBL windows contain only the molecules that
                # were selected for that band.  Keep a typed empty list so
                # the mixture calculation treats that species as zero.
                parsed_window[species] = HitranLines(
                    species=species,
                    molecule_id=MOLECULE_ID_BY_SPECIES[species],
                    isotopologue=np.asarray([], dtype=np.int16),
                    nu0=np.asarray([], dtype=float),
                    strength_ref=np.asarray([], dtype=float),
                    einstein_a=np.asarray([], dtype=float),
                    gamma_air=np.asarray([], dtype=float),
                    gamma_self=np.asarray([], dtype=float),
                    e_lower=np.asarray([], dtype=float),
                    n_air=np.asarray([], dtype=float),
                    delta_air=np.asarray([], dtype=float),
                )
                continue
            parsed_window[species] = HitranLines(
                species=species,
                molecule_id=int(row["molecule_id"][0]),
                isotopologue=np.asarray(row["isotopologue"], dtype=np.int16),
                nu0=np.asarray(row["nu0"], dtype=float),
                strength_ref=np.asarray(row["strength_ref"], dtype=float),
                einstein_a=np.asarray(row["einstein_a"], dtype=float),
                gamma_air=np.asarray(row["gamma_air"], dtype=float),
                gamma_self=np.asarray(row["gamma_self"], dtype=float),
                e_lower=np.asarray(row["e_lower"], dtype=float),
                n_air=np.asarray(row["n_air"], dtype=float),
                delta_air=np.asarray(row["delta_air"], dtype=float),
            )
        parsed[window.name] = parsed_window
    return parsed


def parse_hitran_files_for_windows(
    files_by_species: dict[str, list[Path]],
    windows: list[SpectralWindow],
    min_strength: float = 0.0,
) -> dict[str, dict[str, HitranLines]]:
    """Stream one or more HITRAN/HITEMP files and retain only target windows.

    HITEMP files can be several gigabytes.  The files are therefore scanned
    line-by-line and all three windows are collected in one pass, rather than
    loading a complete file or scanning it once per window.
    """
    records_by_window = {
        window.name: {
            species: _empty_line_records() for species in SPECIES_BY_ID.values()
        }
        for window in windows
    }
    lowest_cm = min(window.start_cm for window in windows)
    highest_cm = max(window.stop_cm for window in windows)
    for species, paths in files_by_species.items():
        if species not in SPECIES_BY_ID.values():
            raise ValueError(f"不支持的分子：{species}")
        for path in paths:
            progress = ProgressBar(
                f"Reading {species} {path.name}", path.stat().st_size, unit="bytes"
            )
            processed_bytes = 0
            with path.open("r", encoding="ascii", errors="strict") as handle:
                for line_number, raw in enumerate(handle, start=1):
                    processed_bytes += len(raw)
                    if line_number % 20000 == 0:
                        progress.update(processed_bytes)
                    line = raw.rstrip("\r\n")
                    if not line or line.startswith("#"):
                        continue
                    if len(line) < 67:
                        raise ValueError(
                            f"{path}:{line_number} has only {len(line)} characters; "
                            "expected a HITRAN fixed-width record"
                        )
                    try:
                        molecule_id = int(line[0:2])
                        nu0 = _as_float(line[3:15])
                        strength_ref = _as_float(line[15:25])
                    except ValueError as exc:
                        raise ValueError(
                            f"无法解析 {path}:{line_number} 的 HITRAN 记录"
                        ) from exc
                    if molecule_id != MOLECULE_ID_BY_SPECIES[species]:
                        continue
                    if strength_ref < min_strength or not lowest_cm <= nu0 <= highest_cm:
                        continue
                    matching_windows = [
                        window
                        for window in windows
                        if window.start_cm <= nu0 <= window.stop_cm
                    ]
                    if not matching_windows:
                        continue
                    values = {
                        "molecule_id": molecule_id,
                        "isotopologue": _isotopologue_id(line[2:3]),
                        "nu0": nu0,
                        "strength_ref": strength_ref,
                        "einstein_a": _as_float(line[25:35]),
                        "gamma_air": _as_float(line[35:40]),
                        "gamma_self": _as_float(line[40:45]),
                        "e_lower": _as_float(line[45:55]),
                        "n_air": _as_float(line[55:59]),
                        "delta_air": _as_float(line[59:67]),
                    }
                    for window in matching_windows:
                        row = records_by_window[window.name][species]
                        for field, value in values.items():
                            row[field].append(value)
            progress.finish()
    return _records_to_lines(records_by_window, windows)


def discover_hitemp_files(
    hitemp_root: Path, windows: list[SpectralWindow]
) -> dict[str, list[Path]]:
    """Find HITEMP files relevant to the requested windows."""
    species_directories = {
        "H2O": hitemp_root / "H2O-2010",
        "CO": hitemp_root / "CO-2019",
        "CH4": hitemp_root / "CH4-2020",
    }
    files_by_species: dict[str, list[Path]] = {}
    for species, directory in species_directories.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"未找到 HITEMP {species} 目录：{directory}")
        candidates = sorted(directory.glob("*.par"))
        if not candidates:
            raise FileNotFoundError(f"HITEMP 目录中没有 .par 文件：{directory}")
        selected: list[Path] = []
        for candidate in candidates:
            labeled_range = _parse_labeled_range(candidate.name)
            if labeled_range is None or any(
                _overlap(labeled_range, window) > 0.0 for window in windows
            ):
                selected.append(candidate)
        if not selected:
            raise FileNotFoundError(
                f"HITEMP {species} 文件没有覆盖目标窗口：{directory}"
            )
        files_by_species[species] = selected
    return files_by_species


def discover_lbl_files(
    lbl_root: Path, windows: list[SpectralWindow]
) -> dict[str, list[Path]]:
    """Select the local LBL file whose labeled directory covers each window."""
    candidates = sorted(lbl_root.rglob("*.par"))
    if not candidates:
        raise FileNotFoundError(f"LBL 目录中没有 .par 文件：{lbl_root}")
    selected_by_window: dict[str, Path] = {}
    for window in windows:
        scored: list[tuple[tuple[int, float, float], Path]] = []
        for candidate in candidates:
            labeled_range = _parse_labeled_range(str(candidate.parent))
            if labeled_range is None:
                continue
            overlap = _overlap(labeled_range, window)
            if overlap <= 0.0:
                continue
            contains = int(
                labeled_range[0] <= window.start_cm
                and labeled_range[1] >= window.stop_cm
            )
            center_distance = abs(
                (labeled_range[0] + labeled_range[1]) / 2.0
                - (window.start_cm + window.stop_cm) / 2.0
            )
            scored.append(((contains, overlap, -center_distance), candidate))
        if not scored:
            raise FileNotFoundError(
                f"LBL 数据没有覆盖窗口 {window.label}：{lbl_root}"
            )
        selected_by_window[window.name] = max(scored, key=lambda item: item[0])[1]

    files_by_species: dict[str, list[Path]] = {
        species: [] for species in SPECIES_BY_ID.values() if species != "NH3"
    }
    for window in windows:
        selected = selected_by_window[window.name]
        for species in files_by_species:
            if selected not in files_by_species[species]:
                files_by_species[species].append(selected)
    return files_by_species


def load_window_line_lists(
    data_source: str,
    line_file: Path | None,
    hitemp_root: Path,
    lbl_root: Path,
    windows: list[SpectralWindow],
    min_strength: float,
    nh3_root: Path | None = None,
) -> tuple[dict[str, dict[str, HitranLines]], dict[str, Any]]:
    """Resolve a data source and load all target windows in one pass."""
    resolved_source = data_source
    if resolved_source == "auto":
        if hitemp_root.is_dir():
            resolved_source = "hitemp"
        elif lbl_root.is_dir():
            resolved_source = "lbl"
        elif line_file is not None and line_file.is_file():
            resolved_source = "line-file"
        else:
            raise FileNotFoundError(
                "auto 数据源未找到 HITEMP、LBL 或 line-file；请显式指定路径"
            )
    if resolved_source == "hitemp":
        files_by_species = discover_hitemp_files(hitemp_root, windows)
    elif resolved_source == "lbl":
        files_by_species = discover_lbl_files(lbl_root, windows)
    elif resolved_source == "line-file":
        if line_file is None or not line_file.is_file():
            raise FileNotFoundError(f"未找到 HITRAN line file：{line_file}")
        files_by_species = {
            species: [line_file] for species in SPECIES_BY_ID.values()
        }
    else:
        raise ValueError(f"未知数据源：{data_source}")
    if resolved_source in ("hitemp", "lbl"):
        nh3_root = nh3_root if nh3_root is not None else lbl_root / "NH3"
        nh3_files = sorted(nh3_root.glob("*.par"))
        if not nh3_files:
            raise FileNotFoundError(f"未找到 NH3 的 HITRAN .par 文件：{nh3_root}")
        files_by_species["NH3"] = nh3_files
    line_lists = parse_hitran_files_for_windows(
        files_by_species, windows, min_strength=min_strength
    )
    for window in windows:
        nh3_lines = line_lists[window.name]["NH3"]
        if not len(nh3_lines):
            raise ValueError(f"窗口 {window.label} 没有 NH3 谱线；请检查 NH3 数据范围与 --min-strength")
        for species_lines in line_lists[window.name].values():
            validate_isotopologues(species_lines)
    manifest = {
        "data_source": resolved_source,
        "nh3_data_source": "line-file" if resolved_source == "line-file" else "local HITRAN LBL",
        "nh3_root": str(nh3_root.resolve()) if nh3_root is not None else None,
        "parameter_names": ["T_top_K", "T_bottom_K", "log10_X_H2O", "log10_X_CO", "log10_X_CH4", "log10_X_NH3"],
        "hitemp_root": str(hitemp_root.resolve()),
        "lbl_root": str(lbl_root.resolve()),
        "line_file": str(line_file.resolve()) if line_file is not None else None,
        "files_by_species": {
            species: [str(path.resolve()) for path in paths]
            for species, paths in files_by_species.items()
        },
        "min_strength_296K": min_strength,
    }
    return line_lists, manifest


def parse_hitran_par(path: Path) -> dict[str, HitranLines]:
    """Parse the standard leading fields of 160-character HITRAN records."""
    records: dict[str, dict[str, list[float]]] = {}
    for species in SPECIES_BY_ID.values():
        records[species] = {
            "molecule_id": [],
            "isotopologue": [],
            "nu0": [],
            "strength_ref": [],
            "einstein_a": [],
            "gamma_air": [],
            "gamma_self": [],
            "e_lower": [],
            "n_air": [],
            "delta_air": [],
        }

    with path.open("r", encoding="ascii") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\r\n")
            if not line:
                continue
            if len(line) != 160:
                raise ValueError(
                    f"{path}:{line_number} has {len(line)} characters; expected 160"
                )
            molecule_id = int(line[0:2])
            if molecule_id not in SPECIES_BY_ID:
                continue
            species = SPECIES_BY_ID[molecule_id]
            row = records[species]
            row["molecule_id"].append(molecule_id)
            row["isotopologue"].append(int(line[2:3]))
            row["nu0"].append(_as_float(line[3:15]))
            row["strength_ref"].append(_as_float(line[15:25]))
            row["einstein_a"].append(_as_float(line[25:35]))
            row["gamma_air"].append(_as_float(line[35:40]))
            row["gamma_self"].append(_as_float(line[40:45]))
            row["e_lower"].append(_as_float(line[45:55]))
            row["n_air"].append(_as_float(line[55:59]))
            row["delta_air"].append(_as_float(line[59:67]))

    parsed: dict[str, HitranLines] = {}
    for species, row in records.items():
        if not row["nu0"]:
            raise ValueError(f"No {species} lines were found in {path}")
        parsed[species] = HitranLines(
            species=species,
            molecule_id=int(row["molecule_id"][0]),
            isotopologue=np.asarray(row["isotopologue"], dtype=np.int16),
            nu0=np.asarray(row["nu0"], dtype=float),
            strength_ref=np.asarray(row["strength_ref"], dtype=float),
            einstein_a=np.asarray(row["einstein_a"], dtype=float),
            gamma_air=np.asarray(row["gamma_air"], dtype=float),
            gamma_self=np.asarray(row["gamma_self"], dtype=float),
            e_lower=np.asarray(row["e_lower"], dtype=float),
            n_air=np.asarray(row["n_air"], dtype=float),
            delta_air=np.asarray(row["delta_air"], dtype=float),
        )
    return parsed


def temperature_profile(t_top: float, t_bottom: float, n_layers: int) -> np.ndarray:
    """Two-parameter profile reconstructed from the documented example."""
    eta = np.linspace(0.0, 1.0, n_layers)
    return t_top + (t_bottom - t_top) * eta**0.65


def _cheb_log_q_ratio(coefficients: np.ndarray, temperature: float | np.ndarray) -> np.ndarray:
    """log(Q(296 K) / Q(T)) from one 15-term Chebyshev fit (T clipped to 296-1800 K)."""
    temperature_clipped = np.clip(np.asarray(temperature, dtype=float), TIPS_T_MIN, TIPS_T_MAX)
    z = 2.0 * (temperature_clipped - TIPS_T_MIN) / (TIPS_T_MAX - TIPS_T_MIN) - 1.0
    return (np.polynomial.chebyshev.chebval(-1.0, coefficients)
            - np.polynomial.chebyshev.chebval(z, coefficients))


def log_partition_function_ratio(
    species: str, temperature: float | np.ndarray, isotopologue: int = 1
) -> np.ndarray:
    """Return ``log(Q(296 K) / Q(T))`` of one isotopologue (TIPS-2021 table fit)."""
    key = (species, int(isotopologue))
    if key not in ISOTOPOLOGUE_LOG_Q_CHEBYSHEV:
        raise KeyError(
            f"No TIPS partition-function fit for {species} isotopologue {isotopologue}; "
            "add it with tools/fit_isotopologue_tips.py"
        )
    return _cheb_log_q_ratio(ISOTOPOLOGUE_LOG_Q_CHEBYSHEV[key], temperature)


def validate_isotopologues(lines: HitranLines) -> None:
    """Fail early (not mid-run) if a line list holds an isotopologue without Q/mass data."""
    unknown = sorted(
        int(i) for i in np.unique(lines.isotopologue)
        if (lines.species, int(i)) not in ISOTOPOLOGUE_INFO
    )
    if unknown:
        raise ValueError(
            f"{lines.species} 谱线含有未登记的同位素编号 {unknown}；"
            "请在 tools/fit_isotopologue_tips.py 中补充质量和配分函数后重新生成 isotopologue_data.py"
        )


def line_molecular_masses(lines: HitranLines) -> np.ndarray:
    """Molar mass in amu of the isotopologue each line belongs to (sets Doppler width)."""
    masses = np.empty(len(lines), dtype=float)
    for iso in np.unique(lines.isotopologue):
        masses[lines.isotopologue == iso] = ISOTOPOLOGUE_INFO[(lines.species, int(iso))][1]
    return masses


def line_log_q_ratio(lines: HitranLines, temperature: float) -> np.ndarray:
    """Per-line log(Q(296)/Q(T)); each line uses its own isotopologue's partition sum."""
    out = np.empty(len(lines), dtype=float)
    for iso in np.unique(lines.isotopologue):
        out[lines.isotopologue == iso] = float(
            log_partition_function_ratio(lines.species, temperature, int(iso))
        )
    return out


def line_strength_at_temperature(lines: HitranLines, temperature: float) -> np.ndarray:
    """HITRAN temperature correction using per-isotopologue TIPS Q ratios."""
    stimulated = (
        -np.expm1(-C2_CM_K * lines.nu0 / temperature)
        / -np.expm1(-C2_CM_K * lines.nu0 / T_REF)
    )
    # HITEMP contains very high lower-state energies; work in log space.
    positive_strength = np.maximum(lines.strength_ref, np.finfo(float).tiny)
    log_strength = (
        np.log(positive_strength)
        + line_log_q_ratio(lines, temperature)
        - C2_CM_K * lines.e_lower * (1.0 / temperature - 1.0 / T_REF)
        + np.log(np.maximum(stimulated, np.finfo(float).tiny))
    )
    return np.exp(np.clip(log_strength, -745.0, 700.0))


def parse_isotopologue_scales(text: str | None) -> dict[tuple[str, int], float]:
    """Parse ``H2O:4=2.0,CH4:3=0.5`` into {(species, isotopologue): factor}.

    HITRAN/HITEMP line strengths already contain the terrestrial isotopic
    abundance, so a factor of 1 means "natural ratio"; 2 doubles that
    isotopologue's lines (e.g. an enhanced D/H or 13C/12C ratio).
    """
    scales: dict[tuple[str, int], float] = {}
    for item in filter(None, (part.strip() for part in (text or "").split(","))):
        match = re.fullmatch(r"([A-Za-z0-9]+):(\d+)=([0-9.eE+-]+)", item)
        if not match:
            raise ValueError(f"同位素权重格式应为 物种:编号=倍数（如 H2O:4=2.0），收到：{item!r}")
        key = (match.group(1), int(match.group(2)))
        if key not in ISOTOPOLOGUE_INFO:
            raise ValueError(f"未登记的同位素 {key[0]}:{key[1]}")
        factor = float(match.group(3))
        if not math.isfinite(factor) or factor < 0.0:
            raise ValueError(f"同位素权重必须是非负有限数：{item!r}")
        scales[key] = factor
    return scales


def apply_isotopologue_scales(
    lines: HitranLines, scales: dict[tuple[str, int], float]
) -> HitranLines:
    """Return a copy of ``lines`` whose 296 K strengths carry the isotopologue weights."""
    factor = np.ones(len(lines), dtype=float)
    for (species, iso), value in scales.items():
        if species == lines.species:
            factor[lines.isotopologue == iso] = value
    if np.all(factor == 1.0):
        return lines
    return replace(lines, strength_ref=lines.strength_ref * factor)


def species_cross_section(
    nu_grid: np.ndarray,
    lines: HitranLines,
    temperature: float,
    pressure_pa: float,
    mole_fraction: float,
    chunk_size: int,
) -> np.ndarray:
    """Return the species cross section in cm^2/molecule on nu_grid."""
    strength = line_strength_at_temperature(lines, temperature)
    molecule_mass = line_molecular_masses(lines) * AMU
    sigma_d = lines.nu0 / C * np.sqrt(K_B * temperature / molecule_mass)
    sigma_d = np.maximum(sigma_d, 1.0e-12)

    pressure_atm = pressure_pa / P_REF_PA
    self_pressure = pressure_atm * mole_fraction
    foreign_pressure = pressure_atm - self_pressure
    gamma_l = (T_REF / temperature) ** lines.n_air * (
        lines.gamma_air * foreign_pressure + lines.gamma_self * self_pressure
    )
    gamma_l = np.maximum(gamma_l, 1.0e-12)
    shifted_center = lines.nu0 + lines.delta_air * foreign_pressure

    cross_section = np.zeros_like(nu_grid)
    sqrt_two = math.sqrt(2.0)
    normalization = math.sqrt(2.0 * math.pi)
    for start in range(0, len(lines), chunk_size):
        stop = min(start + chunk_size, len(lines))
        sd = sigma_d[start:stop, None]
        gl = gamma_l[start:stop, None]
        center = shifted_center[start:stop, None]
        z = ((nu_grid[None, :] - center) + 1j * gl) / (sd * sqrt_two)
        profile = np.real(wofz(z)) / (sd * normalization)
        contribution = np.nan_to_num(
            strength[start:stop, None] * profile,
            nan=0.0,
            posinf=np.finfo(float).max / 1.0e6,
            neginf=0.0,
        )
        cross_section += np.sum(contribution, axis=0)
        cross_section = np.minimum(cross_section, np.finfo(float).max / 1.0e6)
    return cross_section


def opacity_profile(
    nu_grid: np.ndarray,
    p_mid: np.ndarray,
    temperatures: np.ndarray,
    abundances: dict[str, float],
    line_lists: dict[str, HitranLines],
    mean_molecular_mass_amu: float,
    chunk_size: int,
    progress: ProgressBar | None = None,
) -> np.ndarray:
    """Compute mass opacity kappa_nu in m^2/kg for every atmospheric layer."""
    mean_particle_mass = mean_molecular_mass_amu * AMU
    kappa = np.empty((len(p_mid), len(nu_grid)), dtype=float)
    for layer, (pressure, temperature) in enumerate(zip(p_mid, temperatures)):
        mixture_cross_section_cm2 = np.zeros_like(nu_grid)
        for species_index, (species, lines) in enumerate(line_lists.items(), start=1):
            x_species = abundances[species]
            sigma_species = species_cross_section(
                nu_grid,
                lines,
                float(temperature),
                float(pressure),
                x_species,
                chunk_size,
            )
            mixture_cross_section_cm2 += x_species * sigma_species
            if progress is not None:
                progress.update(
                    layer * len(line_lists) + species_index,
                    suffix=f"layer {layer + 1}/{len(p_mid)} {species}",
                )
        # cm^2/molecule -> m^2/kg for a mixture with the stated mean mass.
        kappa[layer] = mixture_cross_section_cm2 * 1.0e-4 / mean_particle_mass
    return kappa


def _weideman_coefficients(term_count: int = 32) -> tuple[float, np.ndarray]:
    """Coefficients for an accurate rational approximation to w(z)."""
    transform_size = 2 * term_count
    k = np.arange(-transform_size + 1, transform_size)
    length_scale = math.sqrt(term_count / math.sqrt(2.0))
    theta = k * math.pi / transform_size
    t = length_scale * np.tan(theta / 2.0)
    values = np.exp(-(t**2)) * (length_scale**2 + t**2)
    values = np.concatenate(([0.0], values))
    coefficients = np.real(np.fft.fft(np.fft.fftshift(values))) / (
        2.0 * transform_size
    )
    return length_scale, coefficients[1 : term_count + 1][::-1].copy()


def create_torch_gpu_workspace(
    line_lists: dict[str, HitranLines],
    nu_grid: np.ndarray,
    p_mid: np.ndarray,
    p_edges: np.ndarray,
    dtype_name: str,
) -> TorchGpuWorkspace:
    """Create one persistent CUDA workspace using the active Python environment."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in the active Python environment") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch CUDA is unavailable; run with the dedicated CUDA environment or use --device cpu"
        )

    device = torch.device("cuda:0")
    dtype = torch.float32 if dtype_name == "float32" else torch.float64
    fields = (
        "strength_ref",
        "gamma_air",
        "gamma_self",
        "e_lower",
        "n_air",
        "delta_air",
    )
    gpu_lines: dict[str, dict[str, Any]] = {}
    spectral_origin = float(np.mean(nu_grid))
    for species, lines in line_lists.items():
        gpu_lines[species] = {
            field: torch.as_tensor(
                getattr(lines, field), dtype=dtype, device=device
            )
            for field in fields
        }
        gpu_lines[species]["isotopologue"] = torch.as_tensor(lines.isotopologue, device=device)
        validate_isotopologues(lines)
        iso_ids = [int(i) for i in np.unique(lines.isotopologue)]
        gpu_lines[species]["iso_ids"] = iso_ids
        gpu_lines[species]["iso_index"] = torch.as_tensor(
            np.searchsorted(np.asarray(iso_ids), lines.isotopologue), dtype=torch.long, device=device)
        gpu_lines[species]["molecular_mass_amu"] = torch.as_tensor(
            line_molecular_masses(lines), dtype=dtype, device=device)
        # HITEMP contains reference strengths far below float32's normal
        # range (some are around 1e-200).  Store log(S296) before the CUDA
        # conversion; converting S296 first would turn weak lines into zero,
        # and clamping that zero would spuriously amplify them.
        gpu_lines[species]["log_strength_ref"] = torch.as_tensor(
            np.log(np.maximum(lines.strength_ref, np.finfo(float).tiny)),
            dtype=dtype,
            device=device,
        )
        gpu_lines[species]["nu0"] = torch.as_tensor(
            lines.nu0, dtype=dtype, device=device
        )
        gpu_lines[species]["nu0_relative"] = torch.as_tensor(
            lines.nu0 - spectral_origin, dtype=dtype, device=device
        )

    length_scale, coefficients = _weideman_coefficients()
    return TorchGpuWorkspace(
        torch=torch,
        device=device,
        dtype=dtype,
        dtype_name=dtype_name,
        device_name=torch.cuda.get_device_name(device),
        torch_version=torch.__version__,
        nu_grid=torch.as_tensor(nu_grid, dtype=dtype, device=device),
        nu_grid_relative=torch.as_tensor(
            nu_grid - spectral_origin, dtype=dtype, device=device
        ),
        p_mid=torch.as_tensor(p_mid, dtype=dtype, device=device),
        p_edges=torch.as_tensor(p_edges, dtype=dtype, device=device),
        line_lists=gpu_lines,
        weideman_l=length_scale,
        weideman_coefficients=torch.as_tensor(
            coefficients, dtype=dtype, device=device
        ),
    )


def _torch_faddeeva(z: Any, workspace: TorchGpuWorkspace) -> Any:
    """Evaluate w(z) on CUDA with stable core and far-wing branches."""
    torch = workspace.torch
    length_scale = workspace.weideman_l
    denominator = length_scale - 1j * z
    transformed = (length_scale + 1j * z) / denominator
    polynomial = torch.zeros_like(z)
    for coefficient in workspace.weideman_coefficients:
        polynomial = polynomial * transformed + coefficient
    rational = (
        2.0 * polynomial / denominator**2
        + (1.0 / math.sqrt(math.pi)) / denominator
    )
    # The rational form loses relative accuracy in float32 when Re[w(z)] is
    # tiny.  The asymptotic series is stable and highly accurate for |z|>10.
    inverse_z2 = 1.0 / (z * z)
    asymptotic_series = (
        1.0
        + 0.5 * inverse_z2
        + 0.75 * inverse_z2**2
        + 1.875 * inverse_z2**3
        + 6.5625 * inverse_z2**4
    )
    asymptotic = (
        1j / (math.sqrt(math.pi) * z) * asymptotic_series
    )
    result = torch.where(torch.abs(z) > 10.0, asymptotic, rational)
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _torch_chebyshev_eval(x: Any, coefficients: Any) -> Any:
    """Evaluate a Chebyshev series using the first-kind recurrence."""
    import torch

    if coefficients.numel() == 0:
        return torch.zeros_like(x)
    t_previous = torch.ones_like(x)
    result = coefficients[0] * t_previous
    if coefficients.numel() == 1:
        return result
    t_current = x
    result = result + coefficients[1] * t_current
    for order in range(2, coefficients.numel()):
        t_next = 2.0 * x * t_current - t_previous
        result = result + coefficients[order] * t_next
        t_previous, t_current = t_current, t_next
    return result


def _torch_log_partition_function_ratio(
    workspace: TorchGpuWorkspace, species: str, iso_ids: list[int], temperatures: Any
) -> Any:
    """CUDA log(Q(296)/Q(T)) for each listed isotopologue; shape [len(iso_ids), batch]."""
    torch = workspace.torch
    temperature_clipped = torch.clamp(temperatures, min=TIPS_T_MIN, max=TIPS_T_MAX)
    z = 2.0 * (temperature_clipped - TIPS_T_MIN) / (TIPS_T_MAX - TIPS_T_MIN) - 1.0
    if not iso_ids:
        return torch.zeros((0, temperatures.shape[-1]), dtype=workspace.dtype, device=workspace.device)
    rows = []
    for iso in iso_ids:
        coefficients = torch.as_tensor(
            ISOTOPOLOGUE_LOG_Q_CHEBYSHEV[(species, iso)],
            dtype=workspace.dtype, device=workspace.device,
        )
        log_q = _torch_chebyshev_eval(z, coefficients)
        log_q_ref = _torch_chebyshev_eval(torch.full_like(z, -1.0), coefficients)
        rows.append(log_q_ref - log_q)
    return torch.stack(rows, dim=0)


def _torch_species_cross_section_batch(
    workspace: TorchGpuWorkspace,
    species: str,
    temperatures: Any,
    pressure_pa: Any,
    mole_fractions: Any,
    chunk_size: int,
) -> Any:
    """CUDA cross sections for a batch of atmospheres at one pressure layer."""
    torch = workspace.torch
    lines = workspace.line_lists[species]
    nu0 = lines["nu0"][:, None]
    temperatures = temperatures[None, :]
    # HITEMP has high lower-state energies.  Evaluate the temperature
    # correction in log space so float32 does not create inf * 0 -> NaN.
    stimulated = (
        -torch.expm1(-C2_CM_K * nu0 / temperatures)
        / -torch.expm1(-C2_CM_K * nu0 / T_REF)
    )
    log_strength_max = math.log(torch.finfo(workspace.dtype).max) - math.log(1.0e6)
    # Each line uses the partition sum of its own isotopologue: evaluate one
    # [iso, batch] table, then gather per line.
    iso_ids = lines["iso_ids"]
    q_table = _torch_log_partition_function_ratio(workspace, species, iso_ids, temperatures[0])
    log_q_ratio = q_table[lines["iso_index"]]
    log_strength = (
        lines["log_strength_ref"][:, None]
        + log_q_ratio
        - C2_CM_K
        * lines["e_lower"][:, None]
        * (1.0 / temperatures - 1.0 / T_REF)
        + torch.log(torch.clamp(stimulated, min=torch.finfo(workspace.dtype).tiny))
    )
    strength = torch.exp(torch.clamp(log_strength, min=-100.0, max=log_strength_max))

    molecule_mass = lines["molecular_mass_amu"][:, None] * AMU
    sigma_d = nu0 / C * torch.sqrt(K_B * temperatures / molecule_mass)
    sigma_d = torch.clamp(sigma_d, min=1.0e-12)

    pressure_tensor = torch.as_tensor(
        pressure_pa, dtype=workspace.dtype, device=workspace.device
    )
    if pressure_tensor.ndim == 0:
        pressure_tensor = pressure_tensor.expand(temperatures.shape[1])
    pressure_atm = pressure_tensor[None, :] / P_REF_PA
    self_pressure = pressure_atm * mole_fractions[None, :]
    foreign_pressure = pressure_atm - self_pressure
    gamma_l = (T_REF / temperatures) ** lines["n_air"][:, None] * (
        lines["gamma_air"][:, None] * foreign_pressure
        + lines["gamma_self"][:, None] * self_pressure
    )
    gamma_l = torch.clamp(gamma_l, min=1.0e-12)
    shifted_center = (
        lines["nu0_relative"][:, None]
        + lines["delta_air"][:, None] * foreign_pressure
    )

    batch_size = temperatures.shape[1]
    cross_section = torch.zeros(
        (batch_size, workspace.nu_grid.numel()),
        dtype=workspace.dtype,
        device=workspace.device,
    )
    sqrt_two = math.sqrt(2.0)
    normalization = math.sqrt(2.0 * math.pi)
    line_count = nu0.shape[0]
    for start in range(0, line_count, chunk_size):
        stop = min(start + chunk_size, line_count)
        sd = sigma_d[start:stop, :, None]
        gl = gamma_l[start:stop, :, None]
        center = shifted_center[start:stop, :, None]
        z = (
            (workspace.nu_grid_relative[None, None, :] - center) + 1j * gl
        ) / (sd * sqrt_two)
        profile = torch.clamp(
            torch.nan_to_num(_torch_faddeeva(z, workspace).real, nan=0.0, posinf=0.0, neginf=0.0),
            min=0.0,
        ) / (
            sd * normalization
        )
        finite_max = torch.finfo(workspace.dtype).max / 1.0e6
        contribution = torch.nan_to_num(
            strength[start:stop, :, None] * profile,
            nan=0.0,
            posinf=finite_max,
            neginf=0.0,
        )
        cross_section += torch.sum(contribution, dim=0)
        cross_section = torch.clamp(
            cross_section,
            min=0.0,
            max=finite_max,
        )
    return cross_section


def torch_opacity_profile_batch(
    workspace: TorchGpuWorkspace,
    temperatures: np.ndarray,
    abundances: dict[str, np.ndarray],
    mean_molecular_mass_amu: float,
    chunk_size: int,
    progress: ProgressBar | None = None,
    merge_layers: bool = False,
) -> Any:
    """Compute [sample, layer, wavenumber] opacity entirely on CUDA."""
    if merge_layers:
        return torch_opacity_profile_batch_merged(
            workspace,
            temperatures,
            abundances,
            mean_molecular_mass_amu,
            chunk_size,
            progress=progress,
        )
    torch = workspace.torch
    temperature_tensor = torch.as_tensor(
        temperatures, dtype=workspace.dtype, device=workspace.device
    )
    abundance_tensors = {
        species: torch.as_tensor(values, dtype=workspace.dtype, device=workspace.device)
        for species, values in abundances.items()
    }
    sample_count, layer_count = temperature_tensor.shape
    kappa = torch.empty(
        (sample_count, layer_count, workspace.nu_grid.numel()),
        dtype=workspace.dtype,
        device=workspace.device,
    )
    mean_particle_mass = mean_molecular_mass_amu * AMU
    with torch.inference_mode():
        for layer in range(layer_count):
            mixture = torch.zeros(
                (sample_count, workspace.nu_grid.numel()),
                dtype=workspace.dtype,
                device=workspace.device,
            )
            pressure = float(workspace.p_mid[layer].item())
            for species_index, species in enumerate(workspace.line_lists, start=1):
                mole_fractions = abundance_tensors[species]
                cross_section = _torch_species_cross_section_batch(
                    workspace,
                    species,
                    temperature_tensor[:, layer],
                    pressure,
                    mole_fractions,
                    chunk_size,
                )
                mixture += mole_fractions[:, None] * cross_section
                if progress is not None:
                    progress.update(
                        layer * len(workspace.line_lists) + species_index,
                        suffix=f"layer {layer + 1}/{layer_count} {species}",
                    )
            mixture = torch.nan_to_num(
                mixture,
                nan=0.0,
                posinf=torch.finfo(workspace.dtype).max / 1.0e6,
                neginf=0.0,
            )
            kappa_layer = mixture * 1.0e-4 / mean_particle_mass
            kappa_max = torch.finfo(workspace.dtype).max / 1.0e6
            kappa[:, layer, :] = torch.clamp(
                torch.nan_to_num(
                    kappa_layer,
                    nan=0.0,
                    posinf=kappa_max,
                    neginf=0.0,
                ),
                min=0.0,
                max=kappa_max,
            )
    return kappa


def torch_opacity_profile_batch_merged(
    workspace: TorchGpuWorkspace,
    temperatures: np.ndarray,
    abundances: dict[str, np.ndarray],
    mean_molecular_mass_amu: float,
    chunk_size: int,
    progress: ProgressBar | None = None,
) -> Any:
    """Compute opacity after flattening the sample and layer dimensions.

    The line-strength, pressure-broadening, and line-profile equations are
    unchanged.  Only the CUDA state batch changes from ``sample`` to
    ``sample * layer``; the returned tensor is reshaped back to
    ``[sample, layer, wavenumber]`` for the two-stream solver.
    """
    torch = workspace.torch
    temperature_tensor = torch.as_tensor(
        temperatures, dtype=workspace.dtype, device=workspace.device
    )
    abundance_tensors = {
        species: torch.as_tensor(values, dtype=workspace.dtype, device=workspace.device)
        for species, values in abundances.items()
    }
    sample_count, layer_count = temperature_tensor.shape
    state_count = sample_count * layer_count
    temperature_flat = temperature_tensor.reshape(state_count)
    pressure_flat = (
        workspace.p_mid[None, :]
        .expand(sample_count, layer_count)
        .reshape(state_count)
    )
    mean_particle_mass = mean_molecular_mass_amu * AMU
    mixture_flat = torch.zeros(
        (state_count, workspace.nu_grid.numel()),
        dtype=workspace.dtype,
        device=workspace.device,
    )
    with torch.inference_mode():
        for species_index, species in enumerate(workspace.line_lists, start=1):
            mole_fraction_flat = (
                abundance_tensors[species][:, None]
                .expand(sample_count, layer_count)
                .reshape(state_count)
            )
            cross_section = _torch_species_cross_section_batch(
                workspace,
                species,
                temperature_flat,
                pressure_flat,
                mole_fraction_flat,
                chunk_size,
            )
            mixture_flat += mole_fraction_flat[:, None] * cross_section
            if progress is not None:
                progress.update(
                    species_index * layer_count,
                    suffix=f"merged layers {species_index}/{len(workspace.line_lists)} {species}",
                )
        kappa_max = torch.finfo(workspace.dtype).max / 1.0e6
        mixture_flat = torch.nan_to_num(
            mixture_flat,
            nan=0.0,
            posinf=kappa_max,
            neginf=0.0,
        )
        kappa_flat = torch.clamp(
            mixture_flat * 1.0e-4 / mean_particle_mass,
            min=0.0,
            max=kappa_max,
        )
    return kappa_flat.reshape(sample_count, layer_count, workspace.nu_grid.numel())


def planck_wavenumber_cm(nu_cm: np.ndarray, temperature: float) -> np.ndarray:
    """Planck radiance per cm^-1: W m^-2 sr^-1 (cm^-1)^-1."""
    nu_m = np.asarray(nu_cm, dtype=float) * 100.0
    exponent = H * C * nu_m / (K_B * temperature)
    b_per_m = 2.0 * H * C**2 * nu_m**3 / np.expm1(exponent)
    return b_per_m * 100.0


def brightness_temperature(nu_cm: np.ndarray, spectral_flux: np.ndarray) -> np.ndarray:
    """Invert F_nu = pi B_nu(T) for the brightness temperature."""
    nu_m = np.asarray(nu_cm, dtype=float) * 100.0
    b_per_m = np.maximum(spectral_flux / math.pi / 100.0, np.finfo(float).tiny)
    numerator = H * C * nu_m / K_B
    denominator = np.log1p(2.0 * H * C**2 * nu_m**3 / b_per_m)
    return numerator / denominator


def two_stream_absorption(
    nu_grid: np.ndarray,
    p_edges: np.ndarray,
    temperatures: np.ndarray,
    kappa_layer_nu: np.ndarray,
    gravity: float,
    mu1: float = 1.0 / math.sqrt(3.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-absorption, LTE two-stream integration."""
    n_layers = len(temperatures)
    dp = np.diff(p_edges)
    dtau = kappa_layer_nu * dp[:, None] / gravity
    transmittance = np.exp(-np.minimum(dtau / mu1, 745.0))

    f_down = np.zeros((n_layers + 1, len(nu_grid)), dtype=float)
    f_up = np.zeros_like(f_down)

    # No incident radiation at the top boundary.
    for layer in range(n_layers):
        source = math.pi * planck_wavenumber_cm(nu_grid, temperatures[layer])
        f_down[layer + 1] = (
            f_down[layer] * transmittance[layer]
            + source * (1.0 - transmittance[layer])
        )

    # Black lower boundary at the bottom-layer temperature.
    f_up[n_layers] = math.pi * planck_wavenumber_cm(nu_grid, temperatures[-1])
    for layer in range(n_layers - 1, -1, -1):
        source = math.pi * planck_wavenumber_cm(nu_grid, temperatures[layer])
        f_up[layer] = (
            f_up[layer + 1] * transmittance[layer]
            + source * (1.0 - transmittance[layer])
        )
    return f_up, f_down, f_up - f_down


def _torch_planck_wavenumber_cm(
    workspace: TorchGpuWorkspace, temperatures: Any
) -> Any:
    """B_nu for temperature vectors; result shape is [sample, wavenumber]."""
    torch = workspace.torch
    nu_m = workspace.nu_grid[None, :] * 100.0
    temperature_column = temperatures[:, None]
    exponent = H * C * nu_m / (K_B * temperature_column)
    b_per_m = 2.0 * H * C**2 * nu_m**3 / torch.expm1(exponent)
    return b_per_m * 100.0


def torch_two_stream_top_flux_batch(
    workspace: TorchGpuWorkspace,
    temperatures: np.ndarray,
    kappa_layer_nu: Any,
    gravity: float,
    mu1: float = 1.0 / math.sqrt(3.0),
) -> Any:
    """Return top upward flux for a CUDA batch without storing every level."""
    torch = workspace.torch
    temperature_tensor = torch.as_tensor(
        temperatures, dtype=workspace.dtype, device=workspace.device
    )
    dp = torch.diff(workspace.p_edges)
    dtau = kappa_layer_nu * dp[None, :, None] / gravity
    transmittance = torch.exp(-torch.clamp(dtau / mu1, max=745.0))
    with torch.inference_mode():
        f_up = math.pi * _torch_planck_wavenumber_cm(
            workspace, temperature_tensor[:, -1]
        )
        for layer in range(temperature_tensor.shape[1] - 1, -1, -1):
            source = math.pi * _torch_planck_wavenumber_cm(
                workspace, temperature_tensor[:, layer]
            )
            f_up = (
                f_up * transmittance[:, layer, :]
                + source * (1.0 - transmittance[:, layer, :])
            )
    return f_up


def torch_two_stream_full(
    workspace: TorchGpuWorkspace,
    temperatures: np.ndarray,
    kappa_layer_nu: Any,
    gravity: float,
    mu1: float = 1.0 / math.sqrt(3.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full single-atmosphere CUDA two-stream output converted back to NumPy."""
    torch = workspace.torch
    temperature_tensor = torch.as_tensor(
        temperatures, dtype=workspace.dtype, device=workspace.device
    )
    dp = torch.diff(workspace.p_edges)
    dtau = kappa_layer_nu * dp[:, None] / gravity
    transmittance = torch.exp(-torch.clamp(dtau / mu1, max=745.0))
    layer_count = temperature_tensor.numel()
    grid_count = workspace.nu_grid.numel()
    f_down = torch.zeros(
        (layer_count + 1, grid_count),
        dtype=workspace.dtype,
        device=workspace.device,
    )
    f_up = torch.zeros_like(f_down)
    with torch.inference_mode():
        for layer in range(layer_count):
            source = math.pi * _torch_planck_wavenumber_cm(
                workspace, temperature_tensor[layer : layer + 1]
            )[0]
            f_down[layer + 1] = (
                f_down[layer] * transmittance[layer]
                + source * (1.0 - transmittance[layer])
            )
        f_up[layer_count] = math.pi * _torch_planck_wavenumber_cm(
            workspace, temperature_tensor[-1:]
        )[0]
        for layer in range(layer_count - 1, -1, -1):
            source = math.pi * _torch_planck_wavenumber_cm(
                workspace, temperature_tensor[layer : layer + 1]
            )[0]
            f_up[layer] = (
                f_up[layer + 1] * transmittance[layer]
                + source * (1.0 - transmittance[layer])
            )
    f_net = f_up - f_down
    return tuple(
        array.detach().cpu().numpy().astype(float, copy=False)
        for array in (f_up, f_down, f_net)
    )


def _error_statistics(
    reference: np.ndarray, calculated: np.ndarray, relative_floor: float = 0.0
) -> dict[str, float]:
    """Absolute and relative error metrics against a reference array."""
    difference = np.asarray(calculated) - np.asarray(reference)
    absolute = np.abs(difference)
    valid_relative = np.abs(reference) > relative_floor
    relative = absolute[valid_relative] / np.abs(reference[valid_relative])
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "max_abs": float(np.max(absolute)),
        "mean_rel_percent": float(100.0 * np.mean(relative)),
        "max_rel_percent": float(100.0 * np.max(relative)),
    }


def calculate_and_save_gpu_error(
    output_dir: Path,
    nu_grid: np.ndarray,
    p_edges: np.ndarray,
    p_mid: np.ndarray,
    temperatures: np.ndarray,
    abundances: dict[str, float],
    line_lists: dict[str, HitranLines],
    mean_molecular_mass_amu: float,
    gravity: float,
    chunk_size: int,
    gpu_kappa: np.ndarray,
    gpu_flux: np.ndarray,
    gpu_brightness_temperature: np.ndarray,
) -> list[str]:
    """Compare one GPU reference atmosphere with the exact CPU/SciPy result."""
    started = time.perf_counter()
    cpu_kappa = opacity_profile(
        nu_grid,
        p_mid,
        temperatures,
        abundances,
        line_lists,
        mean_molecular_mass_amu,
        chunk_size,
    )
    cpu_f_up, _, _ = two_stream_absorption(
        nu_grid, p_edges, temperatures, cpu_kappa, gravity
    )
    cpu_flux = cpu_f_up[0]
    cpu_brightness = brightness_temperature(nu_grid, cpu_flux)

    opacity_floor = max(float(np.max(np.abs(cpu_kappa))) * 1.0e-8, 1.0e-14)
    opacity_stats = _error_statistics(cpu_kappa, gpu_kappa, opacity_floor)
    flux_stats = _error_statistics(cpu_flux, gpu_flux)
    brightness_stats = _error_statistics(cpu_brightness, gpu_brightness_temperature)

    opacity_difference = gpu_kappa - cpu_kappa
    flux_absolute_error = np.abs(gpu_flux - cpu_flux)
    flux_relative_error_percent = 100.0 * flux_absolute_error / np.abs(cpu_flux)
    brightness_absolute_error = np.abs(gpu_brightness_temperature - cpu_brightness)
    save_csv(
        output_dir / "gpu_reference_error.csv",
        [
            "wavenumber_cm-1",
            "cpu_flux",
            "gpu_flux",
            "flux_absolute_error",
            "flux_relative_error_percent",
            "cpu_brightness_temperature_K",
            "gpu_brightness_temperature_K",
            "brightness_temperature_absolute_error_K",
            "opacity_rmse_over_layers_m2_kg-1",
            "opacity_max_absolute_error_over_layers_m2_kg-1",
        ],
        (
            [
                nu_grid[index],
                cpu_flux[index],
                gpu_flux[index],
                flux_absolute_error[index],
                flux_relative_error_percent[index],
                cpu_brightness[index],
                gpu_brightness_temperature[index],
                brightness_absolute_error[index],
                np.sqrt(np.mean(opacity_difference[:, index] ** 2)),
                np.max(np.abs(opacity_difference[:, index])),
            ]
            for index in range(len(nu_grid))
        ),
    )

    report_lines = [
        "GPU numerical error against exact CPU/SciPy reference",
        f"gpu_error_opacity_relative_floor_m2_kg-1={opacity_floor:.8e}",
        f"gpu_error_opacity_mae_m2_kg-1={opacity_stats['mae']:.8e}",
        f"gpu_error_opacity_rmse_m2_kg-1={opacity_stats['rmse']:.8e}",
        f"gpu_error_opacity_max_abs_m2_kg-1={opacity_stats['max_abs']:.8e}",
        f"gpu_error_opacity_mean_rel_percent={opacity_stats['mean_rel_percent']:.8e}",
        f"gpu_error_opacity_max_rel_percent={opacity_stats['max_rel_percent']:.8e}",
        f"gpu_error_flux_mae={flux_stats['mae']:.8e}",
        f"gpu_error_flux_rmse={flux_stats['rmse']:.8e}",
        f"gpu_error_flux_max_abs={flux_stats['max_abs']:.8e}",
        f"gpu_error_flux_mean_rel_percent={flux_stats['mean_rel_percent']:.8e}",
        f"gpu_error_flux_max_rel_percent={flux_stats['max_rel_percent']:.8e}",
        f"gpu_error_brightness_temperature_mae_K={brightness_stats['mae']:.8e}",
        f"gpu_error_brightness_temperature_rmse_K={brightness_stats['rmse']:.8e}",
        f"gpu_error_brightness_temperature_max_abs_K={brightness_stats['max_abs']:.8e}",
        f"gpu_error_cpu_reference_seconds={time.perf_counter() - started:.3f}",
    ]
    (output_dir / "gpu_error_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(report_lines), flush=True)
    return report_lines[1:]


def save_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def save_line_inventory(
    output_dir: Path, line_lists: dict[str, HitranLines]
) -> None:
    rows = []
    for species, lines in line_lists.items():
        has_lines = len(lines) > 0
        rows.append(
            [
                species,
                lines.molecule_id,
                len(lines),
                int(np.min(lines.isotopologue)) if has_lines else "",
                int(np.max(lines.isotopologue)) if has_lines else "",
                float(np.min(lines.nu0)) if has_lines else "",
                float(np.max(lines.nu0)) if has_lines else "",
                float(np.min(lines.strength_ref)) if has_lines else "",
                float(np.max(lines.strength_ref)) if has_lines else "",
            ]
        )
    iso_rows = []
    for species, lines in line_lists.items():
        for iso in np.unique(lines.isotopologue):
            mask = lines.isotopologue == iso
            iso_rows.append([
                species, int(iso), ISOTOPOLOGUE_INFO[(species, int(iso))][0],
                ISOTOPOLOGUE_INFO[(species, int(iso))][1], int(mask.sum()),
                float(np.sum(lines.strength_ref[mask])), float(np.max(lines.strength_ref[mask])),
            ])
    save_csv(
        output_dir / "isotopologue_inventory.csv",
        ["species", "isotopologue_id", "afgl_code", "mass_amu", "line_count",
         "strength_sum_296K", "strength_max_296K"],
        iso_rows,
    )
    save_csv(
        output_dir / "line_inventory.csv",
        [
            "species",
            "hitran_molecule_id",
            "line_count",
            "isotopologue_min",
            "isotopologue_max",
            "nu_min_cm-1",
            "nu_max_cm-1",
            "strength_min_296K",
            "strength_max_296K",
        ],
        rows,
    )


def save_forward_visualization_data(
    output_dir: Path,
    line_lists: dict[str, HitranLines],
    nu_grid: np.ndarray,
    p_edges: np.ndarray,
    p_mid: np.ndarray,
    temperatures: np.ndarray,
    kappa: np.ndarray,
    f_out: np.ndarray,
    t_brightness: np.ndarray,
    progress: ProgressBar,
) -> None:
    """Persist numerical results for the separate visualization entrypoint."""
    line_data = {}
    for species, lines in line_lists.items():
        line_data[f'{species}_nu'] = lines.nu0
        line_data[f'{species}_strength'] = lines.strength_ref
    np.savez_compressed(output_dir / 'line_plot_data.npz', **line_data)
    progress.update(2, suffix='line-list data')
    save_csv(
        output_dir / 'atmosphere_profile.csv',
        ['layer', 'p_top_Pa', 'p_bottom_Pa', 'p_mid_Pa', 'temperature_K'],
        ([layer, p_edges[layer], p_edges[layer+1], p_mid[layer], temperatures[layer]]
         for layer in range(len(temperatures))),
    )
    progress.update(3, suffix='atmosphere CSV')
    save_csv(
        output_dir / 'example_spectrum.csv',
        ['wavenumber_cm-1', 'outgoing_flux_W_m-2_per_cm-1', 'brightness_temperature_K'],
        zip(nu_grid, f_out, t_brightness),
    )
    progress.update(4, suffix='spectrum CSV')
    save_csv(
        output_dir / 'example_opacity_all_layers.csv',
        ['wavenumber_cm-1'] + [f'kappa_layer_{i:02d}_m2_kg-1' for i in range(len(p_mid))],
        ([nu_grid[j], *kappa[:, j]] for j in range(len(nu_grid))),
    )
    progress.update(5, suffix='opacity CSV')


def generate_ml_dataset(
    output_dir: Path,
    sample_count: int,
    seed: int,
    nu_grid: np.ndarray,
    p_edges: np.ndarray,
    p_mid: np.ndarray,
    line_lists: dict[str, HitranLines],
    mean_molecular_mass_amu: float,
    gravity: float,
    chunk_size: int,
    gpu_workspace: TorchGpuWorkspace | None,
    gpu_batch_size: int,
    merge_layers: bool = False,
) -> None:
    if sample_count <= 0:
        return

    parameter_bounds = np.asarray(
        [
            [400.0, 800.0],
            [900.0, 1800.0],
            [-5.5, -2.5],
            [-5.5, -3.0],
            [-7.0, -3.5],
            [-7.0, -3.5],
        ]
    )
    sampler = qmc.LatinHypercube(d=len(parameter_bounds), seed=seed)
    unit_samples = sampler.random(sample_count)
    parameters = qmc.scale(
        unit_samples, parameter_bounds[:, 0], parameter_bounds[:, 1]
    )
    # Keep the documented reference atmosphere as the first sample.
    parameters[0] = [550.0, 1450.0, -3.2, -4.1, -5.0, -5.0]

    log_flux = np.empty((sample_count, len(nu_grid)), dtype=float)
    progress = ProgressBar("ML forward model", sample_count, unit="samples")
    if gpu_workspace is None:
        for sample_index, row in enumerate(parameters):
            t_top, t_bottom, log_h2o, log_co, log_ch4, log_nh3 = row
            temperatures = temperature_profile(t_top, t_bottom, len(p_mid))
            abundances = {
                "H2O": 10.0**log_h2o,
                "CO": 10.0**log_co,
                "CH4": 10.0**log_ch4,
                "NH3": 10.0**log_nh3,
            }
            kappa = opacity_profile(
                nu_grid,
                p_mid,
                temperatures,
                abundances,
                line_lists,
                mean_molecular_mass_amu,
                chunk_size,
            )
            f_up, _, _ = two_stream_absorption(
                nu_grid, p_edges, temperatures, kappa, gravity
            )
            log_flux[sample_index] = np.log10(
                np.maximum(f_up[0], np.finfo(float).tiny)
            )
            progress.update(sample_index + 1, suffix="CPU")
    else:
        eta = np.linspace(0.0, 1.0, len(p_mid)) ** 0.65
        for start in range(0, sample_count, gpu_batch_size):
            stop = min(start + gpu_batch_size, sample_count)
            batch = parameters[start:stop]
            temperatures = (
                batch[:, 0, None]
                + (batch[:, 1] - batch[:, 0])[:, None] * eta[None, :]
            )
            abundances = {
                "H2O": 10.0 ** batch[:, 2],
                "CO": 10.0 ** batch[:, 3],
                "CH4": 10.0 ** batch[:, 4],
                "NH3": 10.0 ** batch[:, 5],
            }
            kappa = torch_opacity_profile_batch(
                gpu_workspace,
                temperatures,
                abundances,
                mean_molecular_mass_amu,
                chunk_size,
                merge_layers=merge_layers,
            )
            f_top = torch_two_stream_top_flux_batch(
                gpu_workspace, temperatures, kappa, gravity
            )
            batch_log_flux = gpu_workspace.torch.log10(
                gpu_workspace.torch.clamp(
                    f_top, min=gpu_workspace.torch.finfo(gpu_workspace.dtype).tiny
                )
            )
            log_flux[start:stop] = (
                batch_log_flux.detach().cpu().numpy().astype(float, copy=False)
            )
            progress.update(stop, suffix=f"GPU batch {stop - start}")

    progress.finish(suffix="complete")

    save_csv(
        output_dir / "ml_inputs.csv",
        ["sample", "T_top_K", "T_bottom_K", "log10_X_H2O", "log10_X_CO", "log10_X_CH4", "log10_X_NH3"],
        ([index, *parameters[index]] for index in range(sample_count)),
    )
    save_csv(
        output_dir / "ml_log_flux.csv",
        ["sample"] + [f"nu_{value:.6f}_cm-1" for value in nu_grid],
        ([index, *log_flux[index]] for index in range(sample_count)),
    )



def resolve_ml_sample_count(cli_value: int | None) -> int:
    """Use a CLI value when provided; otherwise ask the user at run time."""
    if cli_value is None:
        try:
            raw_value = input("请输入要生成的正演样本组数（输入 0 跳过）：").strip()
        except EOFError as exc:
            raise ValueError(
                "未读取到样本组数；非交互运行请使用 --ml-samples N"
            ) from exc
        if not raw_value:
            raise ValueError("正演样本组数不能为空")
        try:
            cli_value = int(raw_value)
        except ValueError as exc:
            raise ValueError("正演样本组数必须是整数") from exc
    if cli_value < 0:
        raise ValueError("正演样本组数不能小于 0")
    return cli_value


def resolve_gpu_error_setting(
    gpu_error_flag: bool, skip_gpu_error_flag: bool
) -> bool:
    """Ask interactively whether the slow CPU reference comparison is wanted."""
    if gpu_error_flag:
        return True
    if skip_gpu_error_flag:
        return False
    try:
        response = input("是否开启 CPU 误差对比？输入 y 开启，直接回车/n 跳过：").strip().lower()
    except EOFError:
        print("未检测到交互输入，默认跳过 CPU 误差对比。", flush=True)
        return False
    return response in {"y", "yes", "1", "是", "启用"}


def choose_gpu_batch_size(
    workspace: TorchGpuWorkspace,
    requested_batch_size: int,
    sample_count: int,
    grid_count: int,
    layer_count: int,
    chunk_size: int,
    target_fraction: float,
) -> int:
    """Choose a conservative batch size targeting a fraction of GPU memory."""
    if requested_batch_size > 0:
        return requested_batch_size
    free_bytes, total_bytes = workspace.torch.cuda.mem_get_info(workspace.device)
    allocated_bytes = total_bytes - free_bytes
    target_bytes = int(total_bytes * target_fraction)
    budget_bytes = max(target_bytes - allocated_bytes, 128 * 1024**2)
    element_size = workspace.torch.tensor(
        [], dtype=workspace.dtype, device=workspace.device
    ).element_size()
    # A temporary complex Voigt tensor plus intermediate/profile buffers are
    # the dominant allocation.  The factor is deliberately conservative.
    bytes_per_sample = max(
        1,
        chunk_size * grid_count * element_size * 8
        + layer_count * grid_count * element_size * 2,
    )
    estimated = max(1, int(budget_bytes / bytes_per_sample))
    selected = min(max(1, sample_count), estimated, 64)
    print(
        f"Auto GPU batch size={selected}; target memory={target_fraction * 100:.0f}% "
        f"({allocated_bytes / 1024**3:.2f} GiB already allocated, "
        f"{total_bytes / 1024**3:.2f} GiB total)",
        flush=True,
    )
    return selected


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    default_hitemp_root = Path(r"H:\dataset\Hitran\HITEMP")
    default_lbl_root = project_dir / "dataset" / "LBL"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--line-file",
        type=Path,
        default=None,
        help="legacy mixed-species HITRAN 160-character .par file",
    )
    parser.add_argument(
        "--data-source",
        choices=("auto", "hitemp", "lbl", "line-file"),
        default="auto",
        help="spectral-line source; auto prefers HITEMP, then local LBL",
    )
    parser.add_argument(
        "--hitemp-root",
        type=Path,
        default=default_hitemp_root,
        help="HITEMP root containing H2O-2010, CO-2019 and CH4-2020",
    )
    parser.add_argument(
        "--lbl-root",
        type=Path,
        default=default_lbl_root,
        help="local LBL root containing one or more labeled .par windows",
    )
    parser.add_argument(
        "--windows",
        default="1200-1400,1900-2300,2800-3200",
        help="comma-separated independent windows in cm-1",
    )
    parser.add_argument(
        "--isotope-scale",
        default="",
        help="isotopologue weights relative to natural abundance, e.g. 'H2O:4=2.0,CH4:3=0.5' "
             "(species:HITRAN local isotopologue id=factor; unlisted isotopologues keep factor 1)",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.1,
        help="spectral grid spacing in cm-1 for the multi-window workflow",
    )
    parser.add_argument(
        "--min-strength",
        type=float,
        default=0.0,
        help="optional 296 K line-strength floor; 0 keeps all selected lines",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / "outputs"
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=None,
        help="legacy override for a single window; otherwise use --grid-step",
    )
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--t-top", type=float, default=550.0)
    parser.add_argument("--t-bottom", type=float, default=1450.0)
    parser.add_argument("--log-h2o", type=float, default=-3.2)
    parser.add_argument("--log-co", type=float, default=-4.1)
    parser.add_argument("--log-ch4", type=float, default=-5.0)
    parser.add_argument("--log-nh3", type=float, default=-5.0)
    parser.add_argument("--nh3-root", type=Path, default=None,
                        help="NH3 HITRAN .par directory, supplements HITEMP or LBL")
    parser.add_argument("--gravity", type=float, default=9.8)
    parser.add_argument("--mean-molecular-mass", type=float, default=28.97)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Compute backend; auto uses CUDA when the active PyTorch supports it",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=0,
        help="Atmospheres evaluated together on CUDA; 0 auto-selects a batch targeting 75%% GPU memory",
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.75,
        help="Target fraction of total CUDA memory for automatic batching",
    )
    parser.add_argument(
        "--merge-layers",
        action="store_true",
        help="Flatten sample and atmospheric-layer states in the CUDA opacity kernel",
    )
    parser.add_argument(
        "--gpu-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="CUDA arithmetic precision; float32 is substantially faster on consumer GPUs",
    )
    parser.add_argument(
        "--gpu-error",
        action="store_true",
        help="Opt in to the slow exact CPU reference comparison after a GPU run",
    )
    parser.add_argument(
        "--skip-gpu-error",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ml-samples",
        type=int,
        default=None,
        help="Number of forward-model samples; omit to enter it interactively, or use 0 to skip",
    )
    parser.add_argument("--seed", type=int, default=20260903)
    return parser


def run_single_window(
    args: argparse.Namespace,
    window: SpectralWindow,
    line_lists: dict[str, HitranLines],
    output_dir: Path,
) -> list[str]:
    args.output_dir = output_dir
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.grid_points is not None:
        nu_grid = np.linspace(window.start_cm, window.stop_cm, args.grid_points)
    else:
        grid_count = int(round((window.stop_cm - window.start_cm) / args.grid_step)) + 1
        nu_grid = np.linspace(window.start_cm, window.stop_cm, grid_count)
    p_edges = np.geomspace(1.0, 1.0e5, args.layers + 1)
    p_mid = np.sqrt(p_edges[:-1] * p_edges[1:])
    temperatures = temperature_profile(args.t_top, args.t_bottom, args.layers)
    abundances = {
        "H2O": 10.0**args.log_h2o,
        "CO": 10.0**args.log_co,
        "CH4": 10.0**args.log_ch4,
        "NH3": 10.0**args.log_nh3,
    }

    gpu_workspace: TorchGpuWorkspace | None = None
    backend_name = "cpu"
    gpu_batch_size = args.gpu_batch_size
    if args.device != "cpu":
        try:
            gpu_workspace = create_torch_gpu_workspace(
                line_lists,
                nu_grid,
                p_mid,
                p_edges,
                args.gpu_dtype,
            )
        except RuntimeError as exc:
            if args.device == "gpu":
                raise SystemExit(f"GPU 初始化失败：{exc}") from exc
            print(f"CUDA unavailable; falling back to CPU: {exc}", flush=True)
        else:
            backend_name = "pytorch-cuda"
            print(
                f"Using GPU: {gpu_workspace.device_name}; "
                f"PyTorch {gpu_workspace.torch_version}; {gpu_workspace.dtype_name}",
                flush=True,
            )
            if args.ml_samples > 0:
                gpu_batch_size = choose_gpu_batch_size(
                    gpu_workspace,
                    args.gpu_batch_size,
                    args.ml_samples,
                    len(nu_grid),
                    len(p_mid),
                    args.chunk_size,
                    args.gpu_memory_fraction,
                )

    print("Computing the documented reference atmosphere...", flush=True)
    reference_progress = ProgressBar(
        "Reference atmosphere",
        len(p_mid) * len(line_lists),
        unit="layer/species",
    )
    reference_progress.update(0, suffix="starting", force=True)
    if gpu_workspace is None:
        kappa = opacity_profile(
            nu_grid,
            p_mid,
            temperatures,
            abundances,
            line_lists,
            args.mean_molecular_mass,
            args.chunk_size,
            progress=reference_progress,
        )
        f_up, f_down, f_net = two_stream_absorption(
            nu_grid, p_edges, temperatures, kappa, args.gravity
        )
    else:
        kappa_gpu = torch_opacity_profile_batch(
            gpu_workspace,
            temperatures[None, :],
            {
                species: np.asarray([mole_fraction])
                for species, mole_fraction in abundances.items()
            },
            args.mean_molecular_mass,
            args.chunk_size,
            progress=reference_progress,
            merge_layers=args.merge_layers,
        )[0]
        f_up, f_down, f_net = torch_two_stream_full(
            gpu_workspace, temperatures, kappa_gpu, args.gravity
        )
        kappa = kappa_gpu.detach().cpu().numpy().astype(float, copy=False)
    reference_progress.finish("complete")
    f_out = f_up[0]
    t_brightness = brightness_temperature(nu_grid, f_out)
    gpu_error_lines: list[str] = []

    if not all(
        np.all(np.isfinite(array))
        for array in [kappa, f_up, f_down, f_net, t_brightness]
    ):
        raise FloatingPointError("Non-finite values occurred in the forward model")
    if np.any(kappa < 0.0) or np.any(f_out <= 0.0):
        raise FloatingPointError("Opacity must be non-negative and flux must be positive")

    if gpu_workspace is not None and args.gpu_error and not args.skip_gpu_error:
        print("Computing exact CPU reference error for the GPU result...", flush=True)
        gpu_error_lines = calculate_and_save_gpu_error(
            args.output_dir,
            nu_grid,
            p_edges,
            p_mid,
            temperatures,
            abundances,
            line_lists,
            args.mean_molecular_mass,
            args.gravity,
            args.chunk_size,
            kappa,
            f_out,
            t_brightness,
        )

    post_progress = ProgressBar("Saving results", 5, unit="steps")
    post_progress.update(0, suffix="starting", force=True)
    save_line_inventory(args.output_dir, line_lists)
    post_progress.update(1, suffix="line inventory")
    save_forward_visualization_data(
        args.output_dir, line_lists, nu_grid, p_edges, p_mid, temperatures,
        kappa, f_out, t_brightness, post_progress,
    )
    post_progress.finish("complete")
    generate_ml_dataset(
        args.output_dir,
        args.ml_samples,
        args.seed,
        nu_grid,
        p_edges,
        p_mid,
        line_lists,
        args.mean_molecular_mass,
        args.gravity,
        args.chunk_size,
        gpu_workspace,
        gpu_batch_size,
        merge_layers=args.merge_layers,
    )

    if gpu_workspace is not None:
        gpu_workspace.torch.cuda.synchronize(gpu_workspace.device)

    elapsed = time.perf_counter() - started
    summary_lines = [
        "HITRAN/HITEMP LBL + pure-absorption two-stream reproduction",
        f"spectral_window={window.label}",
        "existing_npz_or_json_read=false",
        f"grid_cm-1={window.start_cm:g},{window.stop_cm:g},{len(nu_grid)}",
        f"pressure_Pa=1,100000,{args.layers}_layers",
        f"temperature_K={args.t_top},{args.t_bottom}",
        f"log10_abundances={args.log_h2o},{args.log_co},{args.log_ch4}",
        "partition_function=TIPS-2021_chebyshev_degree_14",
        f"partition_function_temperature_range_K={TIPS_T_MIN:g},{TIPS_T_MAX:g}",
        f"line_counts=" + ",".join(f"{key}:{len(value)}" for key, value in line_lists.items()),
        f"kappa_min_max_m2_kg-1={np.min(kappa):.10e},{np.max(kappa):.10e}",
        f"outgoing_flux_min_max={np.min(f_out):.10e},{np.max(f_out):.10e}",
        f"brightness_temperature_min_max_K={np.min(t_brightness):.6f},{np.max(t_brightness):.6f}",
        f"ml_samples_generated={args.ml_samples}",
        f"compute_backend={backend_name}",
        f"compute_device={gpu_workspace.device_name if gpu_workspace is not None else 'CPU'}",
        f"gpu_dtype={gpu_workspace.dtype_name if gpu_workspace is not None else 'not_applicable'}",
        f"gpu_batch_size={gpu_batch_size if gpu_workspace is not None else 'not_applicable'}",
        f"gpu_layer_batching={'merged' if args.merge_layers and gpu_workspace is not None else 'per-layer' if gpu_workspace is not None else 'not_applicable'}",
        f"gpu_error_computed={bool(gpu_error_lines)}",
        *gpu_error_lines,
        f"elapsed_seconds={elapsed:.3f}",
    ]
    (args.output_dir / "run_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(summary_lines), flush=True)
    return summary_lines


def main() -> None:
    validate_physical_constants()
    args = build_parser().parse_args()
    try:
        args.windows = parse_window_specification(args.windows)
        args.ml_samples = resolve_ml_sample_count(args.ml_samples)
        args.gpu_error = resolve_gpu_error_setting(
            args.gpu_error, args.skip_gpu_error
        )
    except ValueError as exc:
        raise SystemExit(f"输入错误：{exc}") from exc
    if args.chunk_size <= 0:
        raise SystemExit("输入错误：--chunk-size 必须大于 0")
    if args.gpu_batch_size < 0:
        raise SystemExit("输入错误：--gpu-batch-size 必须为 0（自动）或大于 0")
    if not 0.1 <= args.gpu_memory_fraction <= 0.95:
        raise SystemExit("输入错误：--gpu-memory-fraction 必须在 0.1 到 0.95 之间")
    if args.grid_points is not None and args.grid_points < 2:
        raise SystemExit("输入错误：--grid-points 必须至少为 2")
    if args.grid_step <= 0.0:
        raise SystemExit("输入错误：--grid-step 必须大于 0")
    if args.min_strength < 0.0:
        raise SystemExit("输入错误：--min-strength 不能小于 0")

    root_output_dir = args.output_dir.resolve()
    root_output_dir.mkdir(parents=True, exist_ok=True)
    windows: list[SpectralWindow] = args.windows
    try:
        line_lists_by_window, data_manifest = load_window_line_lists(
            args.data_source,
            args.line_file,
            args.hitemp_root,
            args.lbl_root,
            windows,
            args.min_strength,
            args.nh3_root,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"数据集加载失败：{exc}") from exc

    try:
        isotope_scales = parse_isotopologue_scales(args.isotope_scale)
    except ValueError as exc:
        raise SystemExit(f"--isotope-scale 参数错误：{exc}") from exc
    if isotope_scales:
        line_lists_by_window = {
            name: {sp: apply_isotopologue_scales(lines, isotope_scales) for sp, lines in per_species.items()}
            for name, per_species in line_lists_by_window.items()
        }
        print("同位素权重（相对天然丰度）：" + ", ".join(
            f"{sp}:{iso}={f:g}" for (sp, iso), f in isotope_scales.items()), flush=True)
    data_manifest["isotopologue_scales"] = {f"{sp}:{iso}": f for (sp, iso), f in isotope_scales.items()}

    manifest = {
        **data_manifest,
        "windows": [
            {
                "name": window.name,
                "start_cm-1": window.start_cm,
                "stop_cm-1": window.stop_cm,
                "grid_step_cm-1": args.grid_step,
                "grid_points_override": args.grid_points,
                "line_counts": {
                    species: len(line_lists_by_window[window.name][species])
                    for species in SPECIES_BY_ID.values()
                },
            }
            for window in windows
        ],
    }
    (root_output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    all_summary_lines: list[str] = [
        "Multi-window HITRAN/HITEMP LBL + two-stream workflow",
        f"data_source={data_manifest['data_source']}",
        f"window_count={len(windows)}",
        f"grid_step_cm-1={args.grid_step}",
        f"ml_samples_generated={args.ml_samples}",
    ]
    for window in windows:
        print(f"\n=== Running spectral window {window.label} ===", flush=True)
        summary_lines = run_single_window(
            args,
            window,
            line_lists_by_window[window.name],
            root_output_dir / window.name,
        )
        all_summary_lines.append(
            f"window={window.name};output_dir={root_output_dir / window.name}"
        )
        all_summary_lines.extend(f"{window.name}.{line}" for line in summary_lines[1:])

    (root_output_dir / "run_summary.txt").write_text(
        "\n".join(all_summary_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(all_summary_lines), flush=True)


if __name__ == "__main__":
    main()
    print("数据已保存；运行 visualize_results.py 生成图像。", flush=True)
