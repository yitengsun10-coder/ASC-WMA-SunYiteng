import csv, json, os, pathlib, subprocess, sys, time

PROJECT = pathlib.Path("/root/autodl-tmp/wma26/unifolm-world-model-action")
MANIFEST = pathlib.Path("/root/autodl-tmp/wma26/asc_repo_repair/ASC-WMA-SunYiteng/amp_20case_benchmark/manifest.csv")
MODE = sys.argv[1]
OUTROOT = PROJECT / "results" / "formal_20case_020" / MODE
OUTROOT.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env["HF_HUB_OFFLINE"] = "1"
env["TRANSFORMERS_OFFLINE"] = "1"
cuda_libs = ":".join(str(p) for p in pathlib.Path("/root/miniconda3/lib/python3.12/site-packages/nvidia").glob("*/lib"))
env["LD_LIBRARY_PATH"] = cuda_libs + ":/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")

rows = list(csv.DictReader(MANIFEST.open()))
summary = []
for index, row in enumerate(rows, 1):
    scenario = row["scenario"]
    case = row["case"]
    caseout = OUTROOT / scenario / case
    caseout.mkdir(parents=True, exist_ok=True)
    log_path = caseout / "run.log"
    timing_path = caseout / "timing.json"
    savedir = caseout / "output"
    cmd = [
        sys.executable, "scripts/evaluation/world_model_interaction.py",
        "--seed", row["seed"], "--ckpt_path", "ckpts/unifolm_wma_dual.ckpt",
        "--config", "configs/inference/world_model_interaction.runtime.yaml",
        "--savedir", str(savedir), "--bs", "1", "--height", row["height"],
        "--width", row["width"], "--unconditional_guidance_scale", "1.0",
        "--ddim_steps", row["ddim_steps"], "--ddim_eta", row["ddim_eta"],
        "--prompt_dir", f"{scenario}/{case}/world_model_interaction_prompts",
        "--dataset", scenario, "--video_length", row["video_length"],
        "--frame_stride", row["frame_stride"], "--n_action_steps", row["n_action_steps"],
        "--exe_steps", row["exe_steps"], "--n_iter", row["n_iter"],
        "--timestep_spacing", row["timestep_spacing"], "--guidance_rescale", row["guidance_rescale"],
        "--perframe_ae", "--amp"
    ]
    if MODE == "baseline":
        cmd += ["--no-compile_diffusion", "--no-cudnn_benchmark"]
    else:
        cmd += ["--no-compile_diffusion", "--cudnn_benchmark",
                "--no-save_intermediate_artifacts", "--no-save_final_tensorboard"]
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    videos = sorted((savedir / "inference").glob("*_full_fs*.mp4"))
    record = {"index": index, "scenario": scenario, "case": case,
              "elapsed_seconds": elapsed, "returncode": proc.returncode,
              "video": str(videos[0]) if len(videos) == 1 else None}
    if proc.returncode == 0 and len(videos) == 1 and videos[0].stat().st_size > 0:
        gt = PROJECT / scenario / case / f"{scenario}_{case}.mp4"
        psnr_path = caseout / "psnr.json"
        psnr_log = caseout / "psnr.log"
        with psnr_log.open("w", encoding="utf-8") as log:
            ps = subprocess.run([
                sys.executable, "psnr_score_for_challenge.py",
                "--gt_video", str(gt), "--pred_video", str(videos[0]),
                "--output_file", str(psnr_path)
            ], cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record["psnr_returncode"] = ps.returncode
        if psnr_path.exists():
            try:
                record["psnr"] = json.loads(psnr_path.read_text())["psnr"]
            except Exception as exc:
                record["psnr_error"] = repr(exc)
    timing_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    summary.append(record)
    (OUTROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)

print(json.dumps({"mode": MODE, "cases": len(summary)}, ensure_ascii=False), flush=True)
