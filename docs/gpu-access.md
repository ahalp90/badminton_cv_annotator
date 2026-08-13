# GPU access

Remote HPC GPU access is available. Several nodes with different GPUs are available.
`engelbart`, `bourbaki` and `carmack` all have access to your home directory.
`engelbart` and `carmack` require you to jump host through `turing`. `engelbart` is
the preferred node for this project as it has the least demand (recommendation from
UNE IT).

Node GPUs are:

- `engelbart` V100 16gb
- `bourbaki` A100 40gb
- `carmack` L40 48gb

## Sutherland for the Issue 38 VLM benchmark

`sutherland` is reached through `turing`:

```bash
ssh turing
ssh sutherland
```

It does not use the same home-directory mapping as the other project hosts.
Do not rely on the checkout, virtual environment, models, or prepared videos
being present there. The benchmark runner stages its restricted inference
snapshot and prepared artifacts to a dedicated local scratch directory instead.

On 12 August 2026, Sutherland had an Nvidia L40 with 46,068 MiB of VRAM,
Apptainer 1.5.3, `tmux`, `flock`, `rsync`, and about 1.51 TB free in
`/scratch`. It did not have `uv`, which is not needed on the remote host after
the local preparation step. `/scratch/cmarti56` did not yet exist; the runner
creates its remote root when staging or setting up.

The interactive two-hop route works even when a direct local `ssh sutherland`
does not forward the required inner credentials. For the benchmark wrapper,
pass an SSH-compatible nested-hop script through `--ssh-command`. That same
script is also exported to `rsync`. Then run the read-only prerequisite check
from the Issue 38 worktree:

```bash
scripts/vlm_scene_benchmark/run_carmack.sh check \
  --remote-host sutherland \
  --remote-root /scratch/cmarti56/issue38-vlm \
  --ssh-command /absolute/path/to/nested-ssh
```

Run the setup and smoke stages only while the GPU is idle. The runner refuses
to start Qwen while another compute process is active. InternVideo3 can be
explicitly allowed to share a GPU, but an exclusive GPU is the reproducible
choice. One earlier check found an Ollama process using 15,024 MiB. Always run
the prerequisite check again instead of assuming that historical state still
applies.

On 13 August 2026, the exclusive InternVideo3 20-minute run completed on
Sutherland in 824.05 seconds. It peaked at 41,079 MiB with BF16 cache and no
CPU offload. This confirms that the pinned InternVideo3 runtime fits the L40.
Its retained predictions scored 25.12% accuracy and 0.0803 macro-F1, so it is
not suitable for integration with the tested prompt.

Sutherland does not change the established Qwen capacity limit. The pinned
complete-shard Qwen3-VL test cannot fit on a 48-GB L40; it needs an 80-GB-class
GPU or a supported multi-GPU host. On 14 August 2026, the separate 10-second
Qwen boundary probe completed on Sutherland. It peaked at 40,831 MiB with BF16
KV cache, no CPU offload, and no swap. It scored zero accuracy and macro-F1.
The short result does not change the whole-shard capacity limit.

## Notes from UNE IT

From a turing terminal you can `ssh -Y engelbart` to get a command prompt on
engelbart. From there, `nvidia-smi` or `nvtop` show the GPU.

The GPU HPC hosts all run Rocky 9 (RHEL 9), which is different from turing's Fedora.
Much software built on turing won't run on the HPC hosts; you need to build it on
engelbart (or another HPC host).

If you require space for large data files, there is a `/scratch` partition on most
HPC hosts. Use it like this:

```bash
mkdir /scratch/comp320a
```

and keep your shared files under that directory. `/scratch` areas are not backed up
but there is 1TB free there. Your home directory only has a 40GB quota. `/scratch`
is also local to the system (not a network filesystem) so it is much faster. You
will need to manage the permissions in `/scratch/comp320a` so that everyone can read
them.

Build your software on engelbart (or bourbaki). If you build it on turing, it will
not work.

Another way of building software is to use apptainer, which is like docker but for
HPC environments. Build an apptainer sandbox, install your python packages and create
a `.sif` file, then use it something like this:

```bash
apptainer exec --nv myproject.sif python train_model.py
```

If you use VS Code for development, you can use its "Remote - SSH" extension. Connect
to `turing.une.edu.au` first, and from there configure it to "jump" to engelbart.
