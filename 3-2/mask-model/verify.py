"""Check the checkpoint checksum and run one forward pass."""
import argparse, hashlib, json
from pathlib import Path
import torch
from infer import DEFAULT_CHECKPOINT, load_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--device', default='auto')
args = parser.parse_args()
metadata = json.loads(DEFAULT_CHECKPOINT.with_suffix('.json').read_text())
actual = hashlib.sha256(DEFAULT_CHECKPOINT.read_bytes()).hexdigest()
if actual != metadata['sha256']:
    raise RuntimeError('Checkpoint checksum mismatch')
model, device, info = load_model(device=args.device)
source = torch.zeros(1, 3, 64, 96, device=device)
with torch.inference_mode():
    output = model(source, source)
if output.shape != (1, 1, 64, 96) or not torch.isfinite(output).all():
    raise RuntimeError('Invalid model output')
print(f'checkpoint_sha256={actual} device={device} finite=True')
print(f'parameters={sum(p.numel() for p in model.parameters())}')
