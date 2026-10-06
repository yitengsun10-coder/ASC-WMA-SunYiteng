import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from run_final_guarded_20case import PROJECT, MANIFEST, BASELINE_SECONDS

root = PROJECT / 'results/formal_20case_020/final_last_20case_v4'
checkpoint = json.loads((root/'inference_complete.json').read_text())
rows = list(csv.DictReader(MANIFEST.open()))
records = checkpoint['records']
assert len(records) == 20
for r in records:
    caseout = root/r['scenario']/r['case']
    psnr = caseout/'psnr.json'
    if not psnr.exists():
        with (caseout/'psnr.log').open('w') as log:
            subprocess.run([sys.executable, 'psnr_score_for_challenge.py', '--gt_video',
                str(PROJECT/r['scenario']/r['case']/f"{r['scenario']}_{r['case']}.mp4"),
                '--pred_video', r['video'], '--output_file', str(psnr)],
                cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
    r['psnr'] = json.loads(psnr.read_text())['psnr']
    probe = subprocess.run(['/root/autodl-tmp/envs/wma26/bin/ffprobe', '-v', 'error',
        '-select_streams','v:0','-count_frames','-show_entries',
        'stream=width,height,avg_frame_rate,nb_read_frames','-of','json',r['video']],
        capture_output=True,text=True,check=True)
    spec = json.loads(probe.stdout)['streams'][0]
    row = rows[r['index']-1]
    r['video_spec'] = spec
    r['video_spec_pass'] = (int(spec['width'])==int(row['width']) and
        int(spec['height'])==int(row['height']) and spec['avg_frame_rate']=='8/1' and
        int(spec['nb_read_frames'])==int(row['expected_frames']))
    (caseout/'timing.json').write_text(json.dumps(r,indent=2)+'\n')
total = checkpoint['inference_total_seconds']
summary = {'status':'pass' if all(r['psnr']>=25 and r['video_spec_pass'] for r in records)
    and BASELINE_SECONDS/total>=1.25 else 'fail', 'cases':20,
    'inference_total_seconds':total, 'amp_baseline_total_seconds':BASELINE_SECONDS,
    'speedup_vs_amp_baseline':BASELINE_SECONDS/total,
    'time_reduction_percent':100*(1-total/BASELINE_SECONDS),
    'minimum_psnr_db':min(r['psnr'] for r in records),
    'mean_case_psnr_db':sum(r['psnr'] for r in records)/20,
    'all_psnr_at_least_25_db':all(r['psnr']>=25 for r in records),
    'all_video_specs_pass':all(r['video_spec_pass'] for r in records),
    'failed_quality_cases':[f"{r['scenario']}/{r['case']}" for r in records if r['psnr']<25],
    'audit_recovery_note':'Original process exit 1: ffprobe missing from PATH after inference completed. CPU-only scoring/spec audit resumed using absolute ffprobe path. Original inference timing unchanged.',
    'records':records}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)
raise SystemExit(0 if summary['status']=='pass' else 2)
