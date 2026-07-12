# CourtKeyNet provenance

Upstream: https://github.com/adithyanraj03/CourtKeyNet (Adithya N Raj), commit f852db65b5d435db16f3c624d2d51dc78b903705, MIT (LICENSE carried in `_vendor/`).

Weights: https://huggingface.co/Cracked-ANJ/CourtKeyNet, revision 8254e285ed0ed496a96fa4632111c80ffb645998, file `finetuned/courtkeynet_finetuned.safetensors`.

Weights sha256: `bcee559a41a54198110120931f7e0aa1d56aa83ab7307e86c4819029346cef57`.

Paper: Hess et al., DOI 10.1016/j.mlwa.2026.100884.

We've used just the model and weights of CourtKeyNet and left off everything non-essential for this project.

Taken verbatim: 
- `courtkeynet/models/*.py` (model definition), 
- `courtkeynet/configs/courtkeynet.yaml` (the `model:` block is what the constructor reads). 

Left behind: 
- GUI `inference.py`, 
- train/finetune scripts, 
- misc: losses, utils, datasets, examples, assets, tools, `.git`.
- author's safetensors.py, which watermarks all weights. Replaced by stock safetensors lib. Originally left off because of some confusion about why the file was obfuscated. We openly attribute court detection to the author of CourtKeyNet.

Input contract (from upstream training dataloader and inference script): 640x640 RGB, squash resize, float32 / 255, no mean/std normalisation. Corner order TL, TR, BR, BL, normalised [0, 1].
