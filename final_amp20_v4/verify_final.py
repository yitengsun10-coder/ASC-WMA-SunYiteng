"""Verify published final evidence without inference or rescore."""
import csv, json, hashlib
from pathlib import Path

root=Path(__file__).resolve().parent
final=root/'final_last_20case_v4'
summary=json.loads((final/'summary.json').read_text())
clock=json.loads((final/'inference_complete.json').read_text())
rows=list(csv.DictReader((root/'manifest.csv').open(encoding='utf-8')))
baseline=json.loads((root/'baseline/summary.json').read_text())
assert len(rows)==len(summary['records'])==len(baseline)==20
videos_present=bool(list(final.rglob('*.mp4')))
if videos_present:assert len(list(final.rglob('*.mp4')))==20
video_hashes={name:digest for digest,name in (line.split(maxsplit=1) for line in (root/'VIDEO_SHA256SUMS.txt').read_text().splitlines())}
assert len(video_hashes)==20
assert clock['completed_cases']==20
total=sum(r['elapsed_seconds'] for r in baseline)
assert total==7830.488590281457 and all(r['returncode']==0 for r in baseline)
scores=[]
for row,rec in zip(rows,summary['records']):
    assert (row['scenario'],row['case'])==(rec['scenario'],rec['case'])
    base=final/row['scenario']/row['case']
    args=json.loads((base/'args.json').read_text())
    score=json.loads((base/'psnr.json').read_text())['psnr']
    assert score==rec['psnr'] and score>=25
    for field in ['seed','width','height','ddim_steps','ddim_eta','video_length','n_action_steps','exe_steps','n_iter','guidance_rescale']:
        assert float(args[field])==float(row[field]),(base,field)
    assert args['amp'] is True and args['save_fps']==8 and args['ddim_eta_after'] is None
    assert args['frame_stride'][0]==int(row['frame_stride'])
    assert args['timestep_spacing']==row['timestep_spacing']
    spec=rec['video_spec']
    assert (spec['width'],spec['height'],spec['avg_frame_rate'],int(spec['nb_read_frames']))==(512,320,'8/1',int(row['expected_frames']))
    assert rec['video_spec_pass'] and rec['returncode']==0
    videos=list((base/'output/inference').glob('*_full_fs*.mp4'))
    expected_paths=[name for name in video_hashes if name.startswith((base/'output/inference').relative_to(root).as_posix()+'/')]
    assert len(expected_paths)==1
    if videos_present:
        assert len(videos)==1 and videos[0].stat().st_size>0
        assert hashlib.sha256(videos[0].read_bytes()).hexdigest()==video_hashes[expected_paths[0]]
    scores.append(score)
assert summary['inference_total_seconds']==clock['inference_total_seconds']
assert summary['speedup_vs_amp_baseline']==total/clock['inference_total_seconds']>=1.25
assert abs(summary['mean_case_psnr_db']-sum(scores)/20)<1e-12
assert summary['minimum_psnr_db']==min(scores)
assert (root/'final_last_20case_v4_exit_code.txt').read_text().strip()=='1'
assert (root/'final_last_20case_v4_audit_exit_code.txt').read_text().strip()=='0'
for line in (root/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
    digest,name=line.split(maxsplit=1)
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
print('PASS: 20 case records and video hashes; all recorded PSNR/specs pass; exact speedup',summary['speedup_vs_amp_baseline'])
print('Full videos present locally:',videos_present,'; Git publication contains metadata and hashes, not videos.')
print('Original wrapper exit 1; CPU recovery audit exit 0; records preserved.')
