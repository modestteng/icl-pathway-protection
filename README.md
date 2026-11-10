# Selective Retention of In-Context Learning through Causal Pathway Protection

Minimal code release for the paper. This repository contains the paper's experiment implementations, necessary configuration, seed identities and upstream source references. It follows the original [icl-dynamics](https://github.com/aadityasingh/icl-dynamics) layout: Python experiment entry points, shared scientific code, explicit random seeds and separately prepared inputs.

The **current tracked tree is limited to 5,000,000 bytes (5 MB)**, including its manifest. Experimental datasets, saved-result arrays/tables, feature files, checkpoints, manuscript files, figures, full logs and development backups are excluded.

## Code and experiments

- `experiments/exp_014_*`, `exp_016_*`, `exp_017_*`, `exp_023_*`: causal interventions and module localization.
- `experiments/exp_030_*` through `exp_038_*`: primary protection, task-loss/random/replay/L2 controls, independent replication and protection-strength experiments.
- `experiments/exp_040_scale_generalization_v2/`: architecture-independent mechanisms, path sets, Adam update projection, and two-layer experiment code.
- `experiments/exp_054_*`, `exp_055_*`: current four-layer adapters. They reuse exp040 kernels and exp047 utilities. Protocol keys, code IDs and generated output directories use the manuscript IDs directly. Only the eight reported four-layer trajectories are configured.
- `experiments/exp_041_*` through `exp_046_*`: frozen external-image evaluation, long-term continuation, training-cost and MEAN25 code.

Formal IDs are `s5` (primary) and `s7` (independent replication). Identify seed IDs together with their experiment group and architecture; the two-layer and four-layer `s18`--`s22` belong to different groups.

[Experiment scope](PAPER_SCOPE.md) identifies each paper family. [Seed identities](reproducibility/SEEDS.md) distinguish initialization, training and experiment-local labels. The [source checksum inventory](reproducibility/source_provenance.json) records SHA-256 hashes for the experiment implementations and shared Python modules.

For the main method, start with `path_set.py` (`set_loss`, `choose_prefix`), `science.py` (`functional_objective`, `modified_step`, `constrain_update`) and `projection.py` under exp040. The current four-layer selector evaluates every ranked prefix of all adjacent-layer Head-to-label-K candidates. Discovery data select paths; held-out confirmation does not reselect them. The four-layer experiments use rho=0.25. Historical two-layer settings retain their original prefix limits.

## Check this source release

```bash
python3 tools/verify_package.py
```

This standard-library check enforces the exact file allowlist, 5 MB budget, per-file hashes, Python/JSON syntax, consistent manuscript/code seed IDs and selected path-selection cases. It imports no scientific modules and runs no training, inference, data encoding or saved-result analysis.

[UPSTREAM_SETUP.md](UPSTREAM_SETUP.md) describes restoring the pinned external model code. CUDA compatibility pins remain in `experiments/reproduction_support/requirements_jax_cuda.txt`. These are the recorded scientific environment, not a newly tested environment.

This is a **source release**, not a complete reproduction archive. Entry points require their separately prepared inputs and, in some cases, historical reference outputs or execution helpers. Historical validators/code manifests remain source records; their omitted inputs are not claimed to be available. Current four-layer adapter manifests enumerate only retained source/configuration. No full experiment rerun or numerical-result validation is claimed for this release.
