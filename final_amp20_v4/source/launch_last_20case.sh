#!/usr/bin/env bash
set -uo pipefail
PROJECT=/root/autodl-tmp/wma26/unifolm-world-model-action
OUTBASE="${PROJECT}/results/formal_20case_020"
TAG=final_last_20case_v4
CUDA_LIBS="$(find /root/miniconda3/lib/python3.12/site-packages/nvidia -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)"
export LD_LIBRARY_PATH="${CUDA_LIBS}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_WAIT_POLICY=PASSIVE
cd "${PROJECT}" || exit 3
test ! -e "${OUTBASE}/${TAG}" || exit 4
date --iso-8601=ns >"${OUTBASE}/${TAG}_start.txt"
/usr/bin/time -f 'full_process_seconds=%e\nexit_status=%x' -o "${OUTBASE}/${TAG}_process_time.txt" \
 /root/autodl-tmp/envs/wma26/bin/python /root/autodl-tmp/wma26/run_clean_20case.py \
 --tag "${TAG}" --isolated-stable-edge >"${OUTBASE}/${TAG}_runner.log" 2>&1
RESULT=$?
printf '%s\n' "${RESULT}" >"${OUTBASE}/${TAG}_exit_code.txt"
date --iso-8601=ns >"${OUTBASE}/${TAG}_end.txt"
exit "${RESULT}"
