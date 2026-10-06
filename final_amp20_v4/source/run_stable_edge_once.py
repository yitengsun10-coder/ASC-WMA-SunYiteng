import time
STARTED = time.perf_counter()
import argparse
import csv
import hashlib
import json
from pathlib import Path
import torch
from pytorch_lightning import seed_everything
from run_final_guarded_20case import PROJECT, MANIFEST, load_module, make_case_args, find_video

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
opts = parser.parse_args()
out = Path(opts.output)
torch.set_num_threads(8)
torch.set_num_interop_threads(2)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
row = next(r for r in csv.DictReader(MANIFEST.open())
    if r['scenario'] == 'unitree_z1_dual_arm_stackbox_v2' and r['case'] == 'case1')
module = load_module()
original_prepare = module.prepare_init_input
active = {}

def prepare_with_fixed_sampling_seed(*args, **kwargs):
    if not active.get('reset'):
        seed_everything(123)
        active['reset'] = True
        active['cpu_rng_sha256'] = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        active['cuda_rng_sha256'] = hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
    return original_prepare(*args, **kwargs)

module.prepare_init_input = prepare_with_fixed_sampling_seed
args = make_case_args(row, out / 'output')
(out / 'args.json').write_text(json.dumps(vars(args), indent=2)+'\n')
seed_everything(123)
runtime = module.run_inference(args, 1, 0, runtime=None)
torch.cuda.synchronize()
video = find_video(Path(args.savedir))
record = {'scenario': row['scenario'], 'case': row['case'], 'video': str(video),
    'returncode': 0, 'worker_seconds_including_imports': time.perf_counter()-STARTED,
    'sampling_rng': active, 'video_sha256': hashlib.sha256(video.read_bytes()).hexdigest()}
(out / 'worker_result.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record), flush=True)
