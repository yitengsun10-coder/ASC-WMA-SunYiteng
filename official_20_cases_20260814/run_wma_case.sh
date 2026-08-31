#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SCENARIO CASE_ID" >&2
  exit 2
fi

SCENARIO=$1
CASE_ID=$2
BASE=/root/autodl-tmp/asc26
PROJ="$BASE/repos/unifolm-world-model-action"
TASK="$BASE/repos/ASC26-Embodied-World-Model-Optimization"
ENV="$BASE/envs/wma26"
CASE="$PROJ/$SCENARIO/$CASE_ID"
RESULT="$BASE/results/wma/$SCENARIO/$CASE_ID"
RUN_LOG="$RESULT/output.log"
TIME_FILE="$RESULT/time.txt"
GPU_LOG="$RESULT/gpu_usage.csv"
STATUS_FILE="$RESULT/status.txt"

mkdir -p "$RESULT"
source "$BASE/environment_paths.sh"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$PROJ"

if [[ ! -d "$CASE" ]]; then
  echo "missing case: $CASE" >&2
  exit 3
fi

CASE_REAL=$(realpath -e "$CASE")
EXPECTED_REAL=$(realpath -e "$PROJ/$SCENARIO/$CASE_ID")
if [[ "$CASE_REAL" != "$EXPECTED_REAL" ]]; then
  echo "refusing unsafe output cleanup: $CASE_REAL" >&2
  exit 5
fi
if [[ -d "$CASE_REAL/output" ]]; then
  rm -rf -- "$CASE_REAL/output"
fi

CSV="$CASE/world_model_interaction_prompts/$SCENARIO.csv"
VIDEO_ID=$("$ENV/bin/python" - "$CSV" <<'PY'
import csv
import sys
with open(sys.argv[1], newline='', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 1:
    raise SystemExit(f"expected one CSV row, found {len(rows)}")
print(rows[0]['videoid'])
PY
)

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 1 > "$GPU_LOG" 2>&1 &
MONITOR_PID=$!
cleanup() { kill "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup EXIT

printf 'RUNNING %s video_id=%s\n' "$(date -Is)" "$VIDEO_ID" > "$STATUS_FILE"
set +e
/usr/bin/time -v -o "$TIME_FILE" \
  bash "$CASE/run_world_model_interaction.sh" > "$RUN_LOG" 2>&1
RC=$?
set -e
cleanup
trap - EXIT

if [[ $RC -ne 0 ]]; then
  printf 'FAILED rc=%s %s video_id=%s\n' "$RC" "$(date -Is)" "$VIDEO_ID" > "$STATUS_FILE"
  exit "$RC"
fi

shopt -s nullglob
candidates=("$CASE/output/inference/${VIDEO_ID}_full_fs"*.mp4)
shopt -u nullglob
if [[ ${#candidates[@]} -ne 1 || ! -s "${candidates[0]}" ]]; then
  printf 'FAILED prediction_count=%s %s video_id=%s\n' "${#candidates[@]}" "$(date -Is)" "$VIDEO_ID" > "$STATUS_FILE"
  exit 4
fi
PRED=${candidates[0]}
GT="$CASE/${SCENARIO}_${CASE_ID}.mp4"

python "$TASK/psnr_score_for_challenge.py" \
  --gt_video "$GT" \
  --pred_video "$PRED" \
  --output_file "$RESULT/psnr_result.json" \
  > "$RESULT/psnr.log" 2>&1

ffprobe -v error -show_entries format=filename,duration,size,bit_rate \
  -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames \
  -of json "$PRED" > "$RESULT/video_probe.json"

cp -a "$RUN_LOG" "$CASE/output.log"
printf 'COMPLETED %s video_id=%s pred=%s\n' "$(date -Is)" "$VIDEO_ID" "$(basename "$PRED")" > "$STATUS_FILE"

