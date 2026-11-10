# Minimal reproduction entry guide

All repository paths below are relative to the repository root. Run commands
from that root. The source release provides the model
interventions and training procedures. A full model rerun requires the registered
external image features, fixed sequence pools, and CUDA environment. The commands
below describe that route; this revision did not rerun encoding or training.

## Check the released records

```bash
python3 tools/verify_package.py
python3 tools/verify_selected_pathways.py
```

The package audit checks the released file hashes, syntax, seed identities, and
selection-rule cases. The pathway checker validates all 18 Table A.1 trajectories against the seed
registry, including group, architecture, ID, initialization seed, training seed, candidate count,
and selected IDs. Where retained depth measurements are present, it also checks
their selected sets. `selected_pathways.json` lists selected IDs in ascending
order, not in discovery rank order.

These are source and metadata checks. They do not train a model, run inference,
or recompute the reported experimental measurements. The public checkout does
not include the author-retained query arrays, checkpoints, or full outputs
needed for an independent saved-result reaggregation.

`reproducibility/selected_pathways.json` provides the complete selected IDs.
`reproducibility/holm_family_record.json` provides the nine historical contrast
definitions and six retained raw/adjusted p-value pairs. Source paths inside
those JSON files identify the author archive; they are provenance locators,
and their raw files are not distributed in this checkout.

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
These are the current four-layer protocols. The reported two-layer architecture
comparison in exp040 used its recorded limit of four prefixes. The current
policy for new experiments is uncapped; re-running that historical comparison
under the new policy is a different experiment.

## Produce the reported summaries

The exp040 aggregation entry is
`experiments/exp_040_scale_generalization_v2/analyze.py`. It reads `ICL040_OUTPUTS`
and writes architecture, seed, contrast, and primary-test summaries when all
expected inputs are present. Its historical analysis requires recorded outputs;
the missing historical four-layer records are not supplied by this source release. The current depth drivers
write batch summaries from their new run roots. Compare newly selected path
sets with `reproducibility/selected_pathways.json` using the experiment group,
architecture, and initialization/training seed pair. New-run outputs remain
in the run roots; the paper's complete measurements are not packaged here.

The historical nine-comparison Holm family is listed in
`reproducibility/holm_family_record.json`. Only six pairs of raw/adjusted p values
remain available; the missing three historical depth records prevent a complete
independent recalculation of that family. The current eight-seed depth test is
a separate analysis.
