"""Probe: what does InpaintNet emit for a fully-missing window?

Run on bourbaki (2026-07-22), where the production checkpoint lives:

    cd ~/badminton_stroke_classification/src/bst_x/TrackNetV3
    PYTHONPATH=. ~/.venvs/venv-bst-x/bin/python probe_inpaint_cycle.py

Feeds the checkpoint the same input every fully-missing non-overlap
window receives at inference: 16 zero coordinate pairs plus an
all-ones fill mask, then applies predict.py:60's int conversion with
the fixture's image scale of 1. Output recorded 2026-07-22:

    seq_len: 16
    x px: [240, 244, 244, 245, 242, 245, 244, 244,
           245, 246, 242, 240, 241, 241, 241, 247]
    y rows: [81, 73, 70, 69, 67, 68, 69, 70,
             68, 68, 71, 73, 77, 80, 83, 84]

probe_vs_track.py (same directory) diffs these against the saved
pilot track.
"""
import torch
from model import InpaintNet

ckpt = torch.load('ckpts/InpaintNet_best.pt', map_location='cpu')
seq_len = ckpt['param_dict']['seq_len']
print('seq_len:', seq_len)

net = InpaintNet()
net.load_state_dict(ckpt['model'])
net.eval()

coords = torch.zeros(1, seq_len, 2)
mask = torch.ones(1, seq_len, 1)
with torch.no_grad():
    out = net(coords, mask)[0]

print('x px:', [int(v * 512) for v in out[:, 0]])
print('y rows:', [int(v * 288) for v in out[:, 1]])
