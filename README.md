# Selective Retention of In-Context Learning through Causal Pathway Protection

Minimal code release for the paper. This repository contains the paper's experiment implementations, necessary configuration, seed definitions and upstream source references. It follows the original [icl-dynamics](https://github.com/aadityasingh/icl-dynamics) layout: Python experiment entry points, shared scientific code, explicit random seeds and separately prepared inputs.

The **current tracked tree is limited to 5,000,000 bytes (5 MB)**, including its manifest. The two manuscript-cited JSON metadata files below are explicitly included. Other experiment-process records, verification reports, large scientific inputs/outputs, manuscripts, figures, logs and development backups are excluded from the Git source tree. The retained research code and data for the manuscript are published separately in the [paper-data release](https://github.com/modestteng/icl-pathway-protection/releases/tag/paper-artifacts-2026-11-10).

## Code and experiments

- `experiments/exp_014_*`, `exp_016_*`, `exp_017_*`, `exp_023_*`: causal interventions and module localization.
- `experiments/exp_030_*` through `exp_038_*`: primary protection, task-loss/random/replay/L2 controls, independent replication and protection-strength experiments.
- `experiments/exp_040_scale_generalization_v2/`: architecture-independent mechanisms, path sets, Adam update projection, and two-layer experiment code.
- `experiments/exp_054_*`, `exp_055_*`: current four-layer adapters. They reuse exp040 kernels and exp047 utilities. Protocol keys, code IDs and generated output directories use the manuscript IDs directly. Only the eight reported four-layer trajectories are configured.
- `experiments/exp_041_*` through `exp_046_*`: frozen external-image evaluation, long-term continuation, training-cost and MEAN25 code.

Formal IDs are `s5` (primary) and `s7` (independent replication). Identify seed IDs together with their experiment group and architecture; the two-layer and four-layer `s18`--`s22` belong to different groups.

[Experiment scope](PAPER_SCOPE.md) identifies each paper family. [Seed identities](reproducibility/SEEDS.md) distinguish initialization, training and experiment-local labels. The [source checksum inventory](reproducibility/source_provenance.json) records SHA-256 hashes for the experiment implementations and shared Python modules.

For the main method, start with `path_set.py` (`set_loss`, `choose_prefix`), `science.py` (`functional_objective`, `modified_step`, `constrain_update`) and `projection.py` under exp040. The exp040 and current four-layer selectors evaluate every ranked prefix of all adjacent-layer Head-to-label-K candidates. Discovery data select paths; held-out confirmation does not reselect them. The four-layer experiments use rho=0.25. The earlier four-prefix cap has also been removed from the exp047 utility driver. Its experimental non-adjacent operator is outside the manuscript's validated adjacent-layer scope.

[Minimal reproduction guide](reproducibility/MINIMAL_REPRODUCTION.md) describes input preparation, released entry points and data download/assembly.

The [paper-data release](https://github.com/modestteng/icl-pathway-protection/releases/tag/paper-artifacts-2026-11-10) contains the retained numerical outputs, available shared features and sequence pools, and four-layer checkpoints. Its manifest and SHA256SUMS identify every file. The paper source snapshot is preserved at the [release tag](https://github.com/modestteng/icl-pathway-protection/tree/paper-artifacts-2026-11-10); the current main branch also includes subsequent uncapped-selector updates.

## Metadata cited by the manuscript

- [Selected pathway IDs](reproducibility/selected_pathways.json) (Appendix A.4 and Table A.1): all 18 trajectories, with experiment group, architecture, initialization/training seeds, complete selected IDs and their indexing convention.
- [Historical Holm family](reproducibility/holm_family_record.json) (Appendix C.2): all nine comparison definitions and the six retained raw/adjusted p-value pairs. The three historical 4L/128 comparisons H1--H3 have no retained sample-size or p-value records and are explicitly marked `unavailable_in_retained_archive` with `null` values. They are missing records, not nonsignificant results. The complete original nine-test Holm correction cannot be independently recalculated from this file. The current eight-pair four-layer test is separate.

These two files are part of the default branch and its exact release allowlist. Paths in their `sources` and `source_file` fields identify the retained research archive; they do not promise that the underlying files are present in the compact Git checkout. Run `python3 tools/verify_selected_pathways.py` to check the 18 pathway/seed identities without scientific imports or model execution.

## Check this source release

```bash
python3 tools/verify_package.py
```

This standard-library check enforces the exact file allowlist, 5 MB budget, per-file hashes, Python/JSON syntax, consistent manuscript/code seed IDs and selected path-selection cases. It imports no scientific modules and runs no training, inference, data encoding or saved-result analysis.

[UPSTREAM_SETUP.md](UPSTREAM_SETUP.md) describes restoring the pinned external model code. CUDA compatibility pins remain in `experiments/reproduction_support/requirements_jax_cuda.txt`. These are the recorded scientific environment, not a newly tested environment.

The Git tree is the compact source release; the linked data release contains all currently retained manuscript research artifacts identified by its manifest. The three historical four-layer Holm-test records are no longer retained and are marked as unavailable in the statistical record. Pinned upstream source and original benchmark datasets remain linked separately. Publishing these files does not constitute a new training or inference rerun.
