# Handover: close the W2 paired-pixel diagnostic

**Status:** complete. No new remote task or full-pool rescore is requested.

Record the false pair's dominant support as a shared raised net band, not two floor markings. Record the approved pair as locally confounded, with white-sock and crossing-sideline positives. Preserve the original whole-candidate rulings: `184:4123` false; `30:33` approved.

Do not adopt connected-run length or any source-pixel sharing flag as an automatic veto. The approved contrary example blocks that interpretation. Do not count missing observations as failures or add these examples to an automatic pool as successes.

When tidying the existing repository outputs, preserve the actual edited `build_pair_atlas.py` whose SHA-256 is `80c0e9c0b97135e08bef4d9f03da2b38b73ddc5b59526d3e2e44800e1a091252`, alongside its returned manifest and evidence. Its source is not in the returned archive. The recorded hashes are sufficient to distinguish this producer from the earlier handover helper. The output directory being inside the repo does not invalidate the run.

`REVIEW.md` is the scientific readout. `results/visual_readout.json` keeps physical interpretations separate from numerical checks. The six tests and standalone verifier have already run in this review. They need no repository imports, model execution or new annotations.
