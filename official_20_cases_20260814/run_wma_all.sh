#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/asc26
RUNNER="$BASE/run_wma_case.sh"
MASTER="$BASE/results/wma/master_status.tsv"

mkdir -p "$(dirname "$MASTER")"
if [[ ! -f "$MASTER" ]]; then
  printf 'scenario\tcase\tstatus\tstarted\tfinished\n' > "$MASTER"
fi

for scenario in \
  unitree_g1_pack_camera \
  unitree_z1_dual_arm_cleanup_pencils \
  unitree_z1_dual_arm_stackbox \
  unitree_z1_dual_arm_stackbox_v2 \
  unitree_z1_stackbox; do
  for case_id in case1 case2 case3 case4; do
    status="$BASE/results/wma/$scenario/$case_id/status.txt"
    if grep -q '^COMPLETED ' "$status" 2>/dev/null; then
      continue
    fi
    started=$(date -Is)
    if "$RUNNER" "$scenario" "$case_id"; then
      rc=COMPLETED
    else
      rc=FAILED
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$scenario" "$case_id" "$rc" "$started" "$(date -Is)" >> "$MASTER"
    if [[ "$rc" == FAILED ]]; then
      exit 1
    fi
  done
done

