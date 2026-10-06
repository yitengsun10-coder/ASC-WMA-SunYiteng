import time
PROCESS_STARTED = time.perf_counter()

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from run_final_guarded_20case import (
    PROJECT, MANIFEST, MODULE_PATH, BASELINE_SECONDS, REQUIRED_SPEEDUP,
    make_case_args, load_module, find_video,
)
import torch
from pytorch_lightning import seed_everything


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='final_clean_20case_v3')
    parser.add_argument('--isolated-stable-edge', action='store_true')
    opts = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    rows = list(csv.DictReader(MANIFEST.open()))
    assert len(rows) == 20
    outroot = PROJECT / 'results/formal_20case_020' / opts.tag
    if outroot.exists() and any(outroot.iterdir()):
        raise RuntimeError(f'Refusing to overwrite {outroot}')
    outroot.mkdir(parents=True, exist_ok=True)
    write_json(outroot / 'run_metadata.json', {
        'policy': 'One generation per case; no candidate selection or retries. Optional fixed isolated deterministic edge path.',
        'isolated_stable_edge': opts.isolated_stable_edge,
        'precision': 'FP16 autocast', 'torch_version': torch.__version__,
        'gpu': torch.cuda.get_device_name(0),
        'torch_threads': torch.get_num_threads(),
        'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        'module_sha256': hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest(),
        'timer': 'perf_counter from first Python statements through all final video writes, before PSNR scoring; Python imports and model initialization included.',
    })
    module = load_module()
    runtime = None
    records = []
    try:
        edge_record = None
        if opts.isolated_stable_edge:
            edgeout = outroot / 'unitree_z1_dual_arm_stackbox_v2' / 'case1'
            edgeout.mkdir(parents=True, exist_ok=True)
            edge_started = time.perf_counter()
            with (edgeout / 'run.log').open('w') as log:
                child = subprocess.run([sys.executable, '/root/autodl-tmp/wma26/run_stable_edge_once.py',
                    '--output', str(edgeout)], cwd=PROJECT, env=os.environ.copy(),
                    stdout=log, stderr=subprocess.STDOUT)
            if child.returncode != 0:
                raise RuntimeError(f'Fixed edge worker failed: {child.returncode}')
            edge_record = json.loads((edgeout / 'worker_result.json').read_text())
            edge_record['elapsed_seconds'] = time.perf_counter() - edge_started
            edge_record['index'] = next(i for i, r in enumerate(rows, 1)
                if (r['scenario'], r['case']) == ('unitree_z1_dual_arm_stackbox_v2', 'case1'))
            records.append(edge_record)
            write_json(edgeout / 'timing.json', edge_record)
            write_json(outroot / 'inference_checkpoint.json', {
                'completed_cases': len(records), 'inference_total_seconds': time.perf_counter()-PROCESS_STARTED,
                'records': records})
        for index, row in enumerate(rows, 1):
            if edge_record is not None and (row['scenario'], row['case']) == ('unitree_z1_dual_arm_stackbox_v2', 'case1'):
                continue
            caseout = outroot / row['scenario'] / row['case']
            caseout.mkdir(parents=True, exist_ok=True)
            args = make_case_args(row, caseout / 'output')
            write_json(caseout / 'args.json', vars(args))
            seed_everything(args.seed)
            started = time.perf_counter()
            runtime = module.run_inference(args, 1, 0, runtime=runtime)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            video = find_video(Path(args.savedir))
            record = {'index': index, 'scenario': row['scenario'], 'case': row['case'],
                      'elapsed_seconds': elapsed, 'returncode': 0, 'video': str(video)}
            records.append(record)
            write_json(caseout / 'timing.json', record)
            write_json(outroot / 'inference_checkpoint.json', {
                'completed_cases': len(records),
                'inference_total_seconds': time.perf_counter() - PROCESS_STARTED,
                'records': records,
            })
            print(json.dumps(record), flush=True)
        torch.cuda.synchronize()
        inference_total = time.perf_counter() - PROCESS_STARTED
        records.sort(key=lambda r: r['index'])
        write_json(outroot / 'inference_complete.json', {
            'completed_cases': 20, 'inference_total_seconds': inference_total,
            'records': records,
        })
        runtime = None
        gc.collect()
        torch.cuda.empty_cache()
        for r in records:
            caseout = outroot / r['scenario'] / r['case']
            gt = PROJECT / r['scenario'] / r['case'] / f"{r['scenario']}_{r['case']}.mp4"
            psnr = caseout / 'psnr.json'
            with (caseout / 'psnr.log').open('w') as log:
                result = subprocess.run([sys.executable, 'psnr_score_for_challenge.py',
                    '--gt_video', str(gt), '--pred_video', r['video'],
                    '--output_file', str(psnr)], cwd=PROJECT,
                    env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0 or not psnr.exists():
                raise RuntimeError(f"Scoring failed: {r['scenario']}/{r['case']}")
            r['psnr_returncode'] = 0
            r['psnr'] = json.loads(psnr.read_text())['psnr']
            probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-count_frames', '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames',
                '-of', 'json', r['video']], capture_output=True, text=True, check=True)
            spec = json.loads(probe.stdout)['streams'][0]
            manifest_row = rows[r['index']-1]
            r['video_spec'] = spec
            r['video_spec_pass'] = (int(spec['width']) == int(manifest_row['width']) and
                int(spec['height']) == int(manifest_row['height']) and
                spec['avg_frame_rate'] == '8/1' and
                int(spec['nb_read_frames']) == int(manifest_row['expected_frames']))
            write_json(caseout / 'timing.json', r)
            print(json.dumps({'scored': r}), flush=True)
        speedup = BASELINE_SECONDS / inference_total
        summary = {
            'status': 'pass' if all(r['psnr'] >= 25 and r['video_spec_pass'] for r in records) and speedup >= REQUIRED_SPEEDUP else 'fail',
            'mode': 'fixed_isolated_edge_plus_persistent_19_no_retries' if opts.isolated_stable_edge else 'single_persistent_20case_no_retries', 'cases': 20,
            'successful_cases': 20, 'inference_total_seconds': inference_total,
            'workflow_seconds_including_scoring': time.perf_counter() - PROCESS_STARTED,
            'amp_baseline_total_seconds': BASELINE_SECONDS,
            'speedup_vs_amp_baseline': speedup,
            'time_reduction_percent': 100*(1-inference_total/BASELINE_SECONDS),
            'minimum_psnr_db': min(r['psnr'] for r in records),
            'mean_case_psnr_db': sum(r['psnr'] for r in records)/len(records),
            'all_video_specs_pass': all(r['video_spec_pass'] for r in records),
            'all_psnr_at_least_25_db': all(r['psnr'] >= 25 for r in records),
            'failed_quality_cases': [f"{r['scenario']}/{r['case']}" for r in records if r['psnr'] < 25],
            'records': records,
        }
        write_json(outroot / 'summary.json', summary)
        print(json.dumps(summary), flush=True)
        return 0 if summary['status'] == 'pass' else 2
    except Exception as exc:
        write_json(outroot / 'failure_checkpoint.json', {
            'error': repr(exc), 'completed_cases': len(records),
            'elapsed_since_process_start_seconds': time.perf_counter()-PROCESS_STARTED,
            'records': records,
        })
        raise


if __name__ == '__main__':
    raise SystemExit(main())
