# Initialization and training seeds / 初始化与训练种子

Each trajectory has one experiment-local ID, shared by the manuscript and code. Protocol keys, seed IDs, output directory names and saved-result metadata use that same ID. Initialization and training numbers are reproducibility parameters.

The machine-readable source is [manuscript_seed_registry.json](manuscript_seed_registry.json). Read a label together with its experimental group and architecture: `s18`--`s22` identify different numerical seed pairs in the two-layer fixed-start and four-layer experiments. Both two-layer architectures share the five pairs in `two_layer_scale`, yielding 25 architecture trajectories from 20 seed-pair records.

## Experiment IDs and random values

| Experimental group | Architecture | ID | Initialization seed | Training seed |
| --- | --- | --- | ---: | ---: |
| primary | 2L/64 | s5 | 5 | 2 |
| independent_replication | 2L/64 | s7 | 7 | 4 |
| two_layer_scale | 2L/64, 2L/128 | s13 | 13 | 10 |
| two_layer_scale | 2L/64, 2L/128 | s14 | 14 | 11 |
| two_layer_scale | 2L/64, 2L/128 | s15 | 15 | 12 |
| two_layer_scale | 2L/64, 2L/128 | s16 | 16 | 13 |
| two_layer_scale | 2L/64, 2L/128 | s17 | 17 | 14 |
| fixed_start_and_transfer | 2L/64 | s18 | 18 | 15 |
| fixed_start_and_transfer | 2L/64 | s19 | 19 | 16 |
| fixed_start_and_transfer | 2L/64 | s20 | 20 | 17 |
| fixed_start_and_transfer | 2L/64 | s21 | 21 | 18 |
| fixed_start_and_transfer | 2L/64 | s22 | 22 | 19 |
| four_layer_depth | 4L/128 | s18 | 23 | 20 |
| four_layer_depth | 4L/128 | s19 | 26 | 23 |
| four_layer_depth | 4L/128 | s20 | 31 | 28 |
| four_layer_depth | 4L/128 | s21 | 33 | 30 |
| four_layer_depth | 4L/128 | s22 | 34 | 31 |
| four_layer_depth | 4L/128 | s23 | 24 | 21 |
| four_layer_depth | 4L/128 | s24 | 25 | 22 |
| four_layer_depth | 4L/128 | s25 | 27 | 24 |

## Group uses and configurations

The primary and independent groups use exp033 and exp036 settings. The two-layer architecture group uses exp040 settings; fixed-start and transfer groups use exp042 settings. The exp054 protocol contains only s18, s19, s23, s24 and s25. The exp055 protocol contains only s20, s21 and s22. Both use the manuscript IDs directly, and together contain exactly the eight four-layer trajectories. No experimental outputs are distributed.

## ID scope and experiment roles

The formal run IDs are `s5` for the primary experiment and `s7` for independent replication; `Primary` and `Replication` are role descriptions only. Identify every trajectory by `(experimental group, architecture, ID)`. In particular, the two-layer fixed-start and four-layer experiments retain the manuscript IDs `s18`--`s22` within their own groups; these denote different numerical seed pairs and must not be mixed.

## Random-state semantics

The initialization seed controls model parameter initialization. The training seed is the root seed for separate data-sampling and model-computation random streams. Continued training restores the checkpoint's model, Adam state and already-advanced random streams without resetting the training seed. Matched A/P25 arms share the complete fork state and primary training-batch order.

## External evaluation data sampling

These seeds construct the fixed image pools and sequences; they do not replace the model initialization/training pairs above.

| Dataset | Support pool | Query pool | Sequence construction |
| --- | ---: | ---: | ---: |
| EMNIST Balanced | 2026104101 | 2026104102 | 2026104103 |
| Kuzushiji-49 | 2026104301 | 2026104302 | 2026104303 |
| CIFAR-100 | 2026104401 | 2026104402 | 2026104403 |

## Verification

```bash
python3 tools/verify_package.py
```

The check compares these experiment IDs and random values with the JSON registry and retained seed configurations. It checks configuration identities only; it does not initialize models or replay random streams.
