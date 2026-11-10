# Minimal reproduction entry guide

All repository paths below are relative to the repository root. Run commands
from that root. The source release provides the model
interventions and training procedures. A full model rerun requires the registered
external image features, fixed sequence pools, and CUDA environment. The commands
below describe that route; this revision did not rerun encoding or training.

## Check the source release

```bash
python3 tools/verify_package.py
```

The audit checks the released file hashes, syntax, seed definitions and synthetic
selection-rule cases. It imports no scientific modules and runs no model.
Experiment notes, verification reports, selected-pathway results and statistical
result records are local only and must not be published.

To validate an author-retained pathway file locally, pass its path explicitly:

```text
python3 tools/verify_selected_pathways.py --record <local-pathway-file.json>
```

The source release does not include the query arrays, checkpoints or outputs
required for saved-result reaggregation.

## Restore the scientific environment and prepare image features

1. Restore the upstream source at the pinned commit with
   `python3 tools/prepare_upstream.py --fetch`. The seven file hashes are in
   `reproducibility/upstream_dependency.json`; restoration does not install dependencies.
2. Reconstruct the CUDA environment using `reproducibility/ENVIRONMENT_GUIDE.txt` and the pins in
   `experiments/exp_040_scale_generalization_v2/input_spec.json`. Image encoding
   uses the separate torch/torchvision pins in
   `experiments/exp_002_main_reproduction/requirements_encoder.txt`.
3. Obtain Omniglot from the source in `DATA_SOURCES.md`. The encoding entry is
   `experiments/exp_002_main_reproduction/extract_features.py`. Its arguments are
   `--source`, `--output`, `--batch-size 64`, and `--data-order-approved` for the
   documented class ordering. It produces `omniglot_features_norotate.h5` and
   `omniglot_features_reordered.h5`. Check the full feature identities against
   `experiments/exp_040_scale_generalization_v2/input_spec.json`; do not replace
   the registered hashes to make an unmatched file pass. A newly encoded file
   with a different identity is a new input reconstruction, not the archived
   experimental input.
4. Create an assets directory with `config.json` copied from
   `experiments/exp_002_main_reproduction/final_00020000000/original_config.json`,
   the two features under `data/`, and the seven restored upstream files under
   `source/`. Use a fresh output directory for new runs.

## Generate the five sequence pools

Set `ICL040_ASSETS` to the prepared assets directory and `ICL040_OUTPUTS` to a
fresh directory. On the CUDA host, run:

```bash
python3 experiments/exp_040_scale_generalization_v2/worker.py --job data
```

The data job uses the frozen sampling configuration and creates
`mechanism_discovery.h5`, `mechanism_confirm.h5`, `protection.h5`, `monitor.h5`,
`confirm.h5`, and `complete.json` under `$ICL040_OUTPUTS/data`. Its audit checks
all ten pool pairs for complete-sequence overlap. The pools share classes and
can reuse individual exemplars.

## Discover pathways and continue training

For the current four-layer cohort, use the uncapped implementation in exp054
and exp055. Each driver enumerates every adjacent-layer head-to-label-K path,
checks every ranked prefix, selects using discovery data at the 50% coverage
target, and uses held-out data for confirmation without reselection. The tested
4L/128 architecture has 24 candidates; this count follows from its structure.

Before starting either driver, create `<run-root>/input_manifest.json` with
`assets` and `data` dictionaries mapping relative filenames to their SHA-256
digests. Register `config.json`, both `data/*.h5` feature files, all seven
`source/*.py` files, all five sequence-pool HDF5 files, and `complete.json`.
The driver verifies these bytes and the pinned source/environment before use.

Run each command below with the prepared absolute paths in place of the angle
brackets. The same paths must be used at every stage:

```text
python3 experiments/exp_054_uncapped_five_seeds/driver.py --root <run-root> --assets <assets> --data <pools> --stage validate --seed s18
python3 experiments/exp_054_uncapped_five_seeds/driver.py --root <run-root> --assets <assets> --data <pools> --stage source --seed s18
python3 experiments/exp_054_uncapped_five_seeds/driver.py --root <run-root> --assets <assets> --data <pools> --stage discover --seed s18
python3 experiments/exp_054_uncapped_five_seeds/driver.py --root <run-root> --assets <assets> --data <pools> --stage arm --seed s18
```

The source stage trains ordinary A and selects the trajectory's monitored ICL
peak. The discover stage ranks paths and selects the set. The arm stage resumes
the common checkpoint with its Adam and random states and trains P25. Both A
and P25 use the same primary-data order; rho is 0.25 in this cohort. Repeat the
source, discover, and arm stages for s19, s23, s24, and s25. For s20--s22, use
the exp055 driver and a separate run root, starting with its validate stage. To summarize each
completed batch, use that driver's `--stage summarize`, then its
`verify_results.py --root <run-root>` command. Logs and complete checkpoints stay
in each run root; `--resume` is for an inspected interrupted source/arm stage.

The seed-to-driver mapping is defined by the released protocol files:
exp054 contains s18, s19, s23, s24, and s25; exp055 contains s20, s21, and s22.
Initialization/training pairs are in `reproducibility/manuscript_seed_registry.json`.
These drivers and the exp040 worker evaluate every ranked prefix of the
complete adjacent-layer candidate pool. Discovery data select paths at the 50%
coverage target; confirmation data do not reselect paths. Use a fresh run root
for each source/configuration identity.

## Produce the reported summaries

The exp040 aggregation entry is
`experiments/exp_040_scale_generalization_v2/analyze.py`. It reads `ICL040_OUTPUTS`
and writes architecture, seed, contrast and primary-test summaries when all
required inputs are present. The depth drivers write batch summaries in their
run roots. Keep all generated summaries, validation reports, logs and other
experiment records outside the published source tree.
