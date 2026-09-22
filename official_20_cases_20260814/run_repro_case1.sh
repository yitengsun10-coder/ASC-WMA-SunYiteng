#!/usr/bin/env bash
set -euo pipefail
BASE=${WMA_BASE:-/root/autodl-tmp/asc26}
PROJ=${WMA_PROJECT:-$BASE/repos/unifolm-world-model-action}
TASK=${WMA_TASK:-$BASE/repos/ASC26-Embodied-World-Model-Optimization}
CASE=unitree_z1_dual_arm_stackbox_v2/case1
OUT=${WMA_OUTPUT_DIR:-$PROJ/$CASE/repro_seed123}
RES=${WMA_RESULT_DIR:-$BASE/results/wma_repro_seed123}
PY=${WMA_PYTHON:-$BASE/envs/wma26/bin/python}
if [[ ! -x "$PY" || ! -d "$PROJ" || ! -f "$TASK/psnr_score_for_challenge.py" ]]; then
  echo "Set WMA_PROJECT, WMA_TASK, and WMA_PYTHON to valid local paths." >&2
  exit 2
fi
mkdir -p "$OUT" "$RES"
cd "$PROJ"
export PATH="$BASE/envs/wma26/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export HF_HOME="$BASE/cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
printf 'RUNNING %s\n' "$(date -Iseconds)" > "$RES/status.txt"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 1 > "$RES/gpu_usage.csv" & MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT
set +e
/usr/bin/time -v -o "$RES/time.txt" "$PY" scripts/evaluation/world_model_interaction.py \
 --seed 123 --ckpt_path ckpts/unifolm_wma_dual.ckpt \
 --config configs/inference/world_model_interaction.yaml --savedir "$OUT" \
 --bs 1 --height 320 --width 512 --unconditional_guidance_scale 1.0 \
 --ddim_steps 50 --ddim_eta 1.0 \
 --prompt_dir "$CASE/world_model_interaction_prompts" \
 --dataset unitree_z1_dual_arm_stackbox_v2 --video_length 16 --frame_stride 4 \
 --n_action_steps 16 --exe_steps 16 --n_iter 11 \
 --timestep_spacing uniform_trailing --guidance_rescale 0.7 --perframe_ae \
 > "$RES/output.log" 2>&1
rc=$?
set -e
kill $MON 2>/dev/null || true; wait $MON 2>/dev/null || true; trap - EXIT
if [ $rc -ne 0 ]; then printf 'FAILED rc=%s %s\n' "$rc" "$(date -Iseconds)" > "$RES/status.txt"; exit $rc; fi
PRED=$(find "$OUT/inference" -maxdepth 1 -type f -name '5_full_fs*.mp4' | head -1)
"$PY" "$TASK/psnr_score_for_challenge.py" \
 --gt_video "$PROJ/$CASE/unitree_z1_dual_arm_stackbox_v2_case1.mp4" \
 --pred_video "$PRED" --output_file "$RES/psnr_result.json" > "$RES/score.log" 2>&1
ffprobe -v quiet -print_format json -show_streams -show_format "$PRED" > "$RES/video_probe.json"
printf 'COMPLETED %s pred=%s\n' "$(date -Iseconds)" "$(basename "$PRED")" > "$RES/status.txt"
