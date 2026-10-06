#!/usr/bin/env python3
"""Fail-closed validator for the required 20-case AMP comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


KEY = ("scenario", "case")
LOCKED = (
    "seed", "width", "height", "ddim_steps", "ddim_eta", "video_length",
    "frame_stride", "n_action_steps", "exe_steps", "n_iter",
    "timestep_spacing", "guidance_rescale", "expected_frames",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def index(rows: list[dict[str, str]], label: str) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = tuple(row.get(name, "") for name in KEY)
        if not all(key) or key in result:
            raise ValueError(f"{label}: missing or duplicate case key {key}")
        result[key] = row
    return result


def finite_float(row: dict[str, str], field: str, label: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label}: non-finite {field}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.csv"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    try:
        manifest = index(read_csv(args.manifest), "manifest")
        baseline = index(read_csv(args.baseline), "baseline")
        optimized = index(read_csv(args.optimized), "optimized")
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 2

    expected = set(manifest)
    if len(expected) != 20:
        errors.append(f"manifest must contain 20 unique cases, found {len(expected)}")
    for name, actual in (("baseline", set(baseline)), ("optimized", set(optimized))):
        if actual != expected:
            errors.append(f"{name} case set mismatch: missing={sorted(expected-actual)} extra={sorted(actual-expected)}")

    baseline_total = 0.0
    optimized_total = 0.0
    optimized_psnr: list[float] = []
    for key in sorted(expected & set(baseline) & set(optimized)):
        name = "/".join(key)
        want, base, opt = manifest[key], baseline[key], optimized[key]
        for field in LOCKED:
            if base.get(field) != want.get(field):
                errors.append(f"baseline {name}: {field}={base.get(field)!r}, expected {want.get(field)!r}")
            if opt.get(field) != want.get(field):
                errors.append(f"optimized {name}: {field}={opt.get(field)!r}, expected {want.get(field)!r}")
        if base.get("eta_policy") != want.get("baseline_eta_policy"):
            errors.append(
                f"baseline {name}: eta_policy={base.get('eta_policy')!r}, "
                f"expected {want.get('baseline_eta_policy')!r}"
            )
        if opt.get("eta_policy") != want.get("optimized_eta_policy"):
            errors.append(
                f"optimized {name}: eta_policy={opt.get('eta_policy')!r}, "
                f"expected {want.get('optimized_eta_policy')!r}"
            )
        for variant, row in (("baseline", base), ("optimized", opt)):
            if row.get("status") != "COMPLETED":
                errors.append(f"{variant} {name}: status={row.get('status')!r}")
            if row.get("precision") != "amp_fp16":
                errors.append(f"{variant} {name}: precision must be amp_fp16")
            if row.get("resolution") != "512x320":
                errors.append(f"{variant} {name}: resolution={row.get('resolution')!r}")
            if row.get("frames") != want.get("expected_frames"):
                errors.append(f"{variant} {name}: frames={row.get('frames')!r}, expected {want.get('expected_frames')!r}")
        try:
            baseline_total += finite_float(base, "elapsed_s", f"baseline {name}")
            optimized_total += finite_float(opt, "elapsed_s", f"optimized {name}")
            psnr = finite_float(opt, "psnr_db", f"optimized {name}")
            optimized_psnr.append(psnr)
            if psnr < 25.0:
                errors.append(f"optimized {name}: PSNR {psnr:.9f} < 25 dB")
        except ValueError as exc:
            errors.append(str(exc))

    speedup = baseline_total / optimized_total if optimized_total > 0 else 0.0
    reduction = (1.0 - optimized_total / baseline_total) * 100.0 if baseline_total > 0 else 0.0
    if speedup < 1.25:
        errors.append(f"total speedup {speedup:.6f}x < 1.25x")

    summary = {
        "cases": len(optimized_psnr),
        "baseline_total_s": baseline_total,
        "optimized_total_s": optimized_total,
        "speedup": speedup,
        "reduction_percent": reduction,
        "optimized_min_psnr_db": min(optimized_psnr) if optimized_psnr else None,
        "optimized_mean_psnr_db": (sum(optimized_psnr) / len(optimized_psnr)) if optimized_psnr else None,
        "pass": not errors,
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
