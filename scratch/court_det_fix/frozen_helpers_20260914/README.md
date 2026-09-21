# Frozen helper scripts of 14 September 2026

Code-only copy, made 2026-09-16, of the `.py` files under the gitignored `scratch/court_det_fix/worklog/checks/independent/player_guided/20260914/` tree that the direction experiment and the line-identity series import (the tree their documents call L), plus `legacy/zone_net.py` and its `camera_diagnostic.py` dependency from the 20260908 tree. Records, images and caches of that tree are not copied. The compute host holds its own copy under the remote experiment root; `../direction_agreement/evidence.md` records that the 43 shared modules were byte-identical between the two on 2026-09-15.

Import roots the experiments put on `sys.path`: `vp_pruning/`, `marking_diagnosis/`, `axis_matching/`, `automatic_axes/`, `automatic_axes/svd_fixed/`, and `legacy/` for `zone_net`.

| File | Bytes | MD5 |
| --- | ---: | --- |
| `automatic_axes/bound_counts.py` | 1,636 | `26f3cdf9d22dc3765bb4d137ce693d6f` |
| `automatic_axes/check_pool_replays.py` | 1,611 | `41e279963977d25bc2a37f8bca36a412` |
| `automatic_axes/check_replay.py` | 1,209 | `d90e18d8e77f088343e1e675f3edceb1` |
| `automatic_axes/diagnose_automatic.py` | 5,540 | `a2a6c0ec0994790c200c71fa5e29aab2` |
| `automatic_axes/diagnose_direction_bank.py` | 7,819 | `4c2c02bd4a93b4df91a635a70b4249ed` |
| `automatic_axes/diagnose_directions.py` | 3,255 | `5ffe3d4ff4baaf5ff5cd4dd8a369102b` |
| `automatic_axes/gx0_control/collected/diagnose_gx0.py` | 2,653 | `f64490235b1e002e23bafdc1079c6995` |
| `automatic_axes/gx0_control/diagnose_gx0.py` | 2,653 | `f64490235b1e002e23bafdc1079c6995` |
| `automatic_axes/gx0_control/render_control.py` | 3,286 | `222a13b9904089164bfd0d89b8184aa2` |
| `automatic_axes/render_automatic_check.py` | 11,597 | `9629c3bcdb799c1eb353b72173510359` |
| `automatic_axes/rescore_camera_pool.py` | 5,650 | `7c65dd04c235979e4f117500ccf08262` |
| `automatic_axes/run_automatic.py` | 9,065 | `e5287191ff275d383e02cbc6d8a4b034` |
| `automatic_axes/run_automatic_initial.py` | 7,739 | `93cb74f165dfbe90b9fda10e06cae347` |
| `automatic_axes/run_bank_control.py` | 3,027 | `a46434a5d4985e273c0996e9201bd7f3` |
| `automatic_axes/svd_fixed/run_svd_fixed.py` | 9,145 | `ca28990becaedc52faa8fad9a2a97c59` |
| `automatic_axes/svd_fixed/test_svd_fixed.py` | 3,132 | `50fb70cd9d036fe5b95b115953d6684d` |
| `automatic_axes/test_automatic.py` | 3,357 | `90fcacc0399b9d2ed863ebd85cc71c28` |
| `automatic_axes/test_camera_pool.py` | 2,709 | `db52ebe040ec4b454f4a3eec075206ab` |
| `automatic_axes/webui_packet/export_evidence.py` | 7,878 | `9d4d15d246e47ae02a1e22068e03d3a7` |
| `automatic_axes/webui_packet/export_temporal.py` | 5,742 | `d55361445307f2ccf3276abcf4a533c2` |
| `axis_matching/collected/projective_seed.py` | 10,449 | `afeffffa08c7a0be64c24bc9bf9c80da` |
| `axis_matching/collected/run_given.py` | 11,160 | `ae9026bad44e55e899718262c6db1de2` |
| `axis_matching/collected/test_projective_seed.py` | 4,344 | `fd0fe715743befa87e407ee30cef9a95` |
| `axis_matching/diagnose_aliases.py` | 6,414 | `541da86d3c78ae96c746ac798711c304` |
| `axis_matching/inspect_appearance.py` | 5,973 | `1df7afd8e59ad019f20d92555162fbcb` |
| `axis_matching/projective_seed.py` | 10,811 | `4cdb8e863318d5a8982a56f68005a1dc` |
| `axis_matching/render_axis_check.py` | 5,609 | `4b3c3d3342df3c6c7666fc965592b877` |
| `axis_matching/run_given.py` | 13,684 | `5df2ee73f71ea408e1c5567fc6f61c8c` |
| `axis_matching/test_projective_seed.py` | 6,387 | `0f086767f21cff70a3960875a6cf6073` |
| `legacy/camera_diagnostic.py` | 2,579 | `fc3472445860c212a7ac9b477c72df79` |
| `legacy/zone_net.py` | 8,325 | `de55928ecc99f62480e9dc43885b8f6b` |
| `marking_diagnosis/render_followup.py` | 6,264 | `b5f1464c550f6870d2ece83e0581d5d1` |
| `marking_diagnosis/run_diagnosis.py` | 12,133 | `1bc56f6bd592e6cb49b687e51d2bf7bf` |
| `marking_diagnosis/scan_population.py` | 7,703 | `deeef56f4983d0b5e72f8a647ef6d080` |
| `marking_diagnosis/summarise.py` | 3,442 | `b43cf6f962c8b3207fd1da0228733db0` |
| `marking_diagnosis/test_diagnosis.py` | 3,393 | `e9e88e5d6b869b57b4b28206271faa53` |
| `marking_diagnosis/verify_leads.py` | 4,426 | `1551144b6a18561ec8174ce59f0f9523` |
| `vp_pruning/diagnose_targets.py` | 5,628 | `ac819d86fa4dc1af021898e216e5f49a` |
| `vp_pruning/render_viewer.py` | 6,761 | `553447ec4af18f052f4ef4c7d9c7f108` |
| `vp_pruning/run_population.py` | 3,968 | `1bad3dff5c3cfa37883a71bb5bdbb5b8` |
| `vp_pruning/run_score.py` | 8,354 | `4f118667fa58e7c89c0c4f661be56ff5` |
| `vp_pruning/test_vp_pruning.py` | 6,344 | `a59e0b92c4e71b7adbac0b11d82aca57` |
| `vp_pruning/vp_pruning.py` | 11,436 | `0fc8080c36157da21949994fec4862e8` |

The direction experiment's own `tests/conftest.py` and `render_gallery.py` still name the gitignored tree, because that experiment's folder is frozen; point its `paths.local.sh` alias `LOCAL_L` at this folder to run them from a clone (the gallery also needs baseline all-camera records that are not copied here).
