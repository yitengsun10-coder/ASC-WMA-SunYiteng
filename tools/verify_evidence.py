#!/usr/bin/env python3
"""Verify the final 20-case WMA acceptance CSV using only the standard library."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    csv_path = args.root / "official_20_cases_20260814" / "wma_20case_final_summary.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")))
    failures = []
    if len(rows) != 20:
        failures.append(f"expected 20 cases, found {len(rows)}")

    psnrs = []
    for row in rows:
        label = f"{row.get('scenario')}/{row.get('case')}"
        try:
            psnr = float(row["psnr_db"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"invalid PSNR for {label}")
            continue
        psnrs.append((label, psnr))
        if row.get("status") != "COMPLETED":
            failures.append(f"{label}: status={row.get('status')}")
        if psnr < 25.0 or row.get("psnr_ge_25", "").lower() != "true":
            failures.append(f"{label}: PSNR={psnr}")

    if psnrs:
        lowest = min(psnrs, key=lambda item: item[1])
        mean = sum(value for _, value in psnrs) / len(psnrs)
        print(f"cases={len(rows)} mean_psnr={mean:.12f} dB min={lowest[1]:.12f} dB ({lowest[0]})")
    if failures:
        print("FAIL:", "; ".join(failures))
        return 1
    print("PASS: 20/20 cases are COMPLETED and 20/20 have PSNR >= 25 dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
