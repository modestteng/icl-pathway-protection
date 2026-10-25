# Selective Retention of In-Context Learning through Causal Pathway Protection

This code and saved-result package contains only experiments reported in the bound manuscript and their required model, data-preparation, statistical, and plotting dependencies. The bound version has 8 numbered figures, 18 tables, and Algorithm 1. See [PAPER_SCOPE.md](PAPER_SCOPE.md) for the experiment-to-paper mapping and `reproducibility/manuscript_binding.json` for the source SHA-256.

## Check the package

```bash
python3 tools/verify_package.py
python3 tools/check_baseline_curve.py
python3 tools/reproduce_paper_results.py --output /tmp/icl-paper-check.json
```

The first check uses the Python standard library. The saved-array check also needs NumPy; review dependencies are in `requirements-review.txt`. The optional `tools/reproduce_section5.py` requires ReportLab and recreates the numerical Section 5 plots from saved data.

For training or model evaluation, restore the original upstream source separately using [UPSTREAM_SETUP.md](UPSTREAM_SETUP.md). The public repository contains this project's experiment scripts and saved results; the seven upstream implementation files and their frozen duplicates are external dependencies. Saved-result checks work without those source files.

## Contents

The release includes the primary behavioral baseline, module replacement and label-K validation, main protection and control methods including MEAN25, the reported s7 replication, protection-strength comparisons, architecture experiments, long-term continuations, frozen external-image evaluations, and four-round training-cost measurements. Primary and later supplementary cohorts remain identified separately. Mixed and negative outcomes of reported experiments are retained.

Model weights, image databases, feature HDF5 files, and complete production logs are external assets. Their identities remain documented, but they are not bundled. The checks above validate the released files and reaggregate selected saved results; they do not rerun training, model forwards, image encoding, or GPU timings. Historical production verifiers remain bound to their original sealed inputs and hashes; release path/scope adaptations do not establish a successful full rerun.

## Paths and provenance

Private filesystem prefixes and operational metadata have been removed. Asset references are portable names, not download promises. `ICL_EXPERIMENT_ROOT` can identify a separately restored experiment tree for source entry points that previously used a private remote root. Retained numeric results are selected without changing their scientific values. `reproducibility/release_changes.json` identifies original and released hashes for transformed files.

The public repository starts from a new release commit and excludes the earlier private development history. The release files contain no author biographies/photos, contact details, account identifiers, private host configuration, research diary, or unrelated experiment results. Public upstream software and dataset attribution remains in `THIRD_PARTY_NOTICES.txt` and `DATA_SOURCES.md`.
