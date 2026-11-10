# Pinned upstream source

The model, sampler, interventions and Omniglot preparation code are external dependencies from [aadityasingh/icl-dynamics](https://github.com/aadityasingh/icl-dynamics), pinned to commit `85b895c720844e795734b9391b32d2619065f9a1`. File identities are in `reproducibility/upstream_dependency.json`. The original seven files are not redistributed here.

To restore only those seven pinned source files and their required local copies:

```bash
python3 tools/prepare_upstream.py --fetch
```

Alternatively, use an existing upstream checkout:

```bash
python3 tools/prepare_upstream.py --source-dir ../icl-dynamics
```

The helper verifies every hash before writing and refuses to overwrite different files. It downloads no datasets or model weights and executes no fetched source. Generated source copies are ignored by Git and excluded from the 5 MB release budget.

Scientific entry points also need external feature/checkpoint/data files in the locations their configuration specifies. For exp054/055, the driver accepts `--root`, `--assets`, `--data`, `--stage` and `--seed`; the runtime root must have the experiment's `input_manifest.json` and the exact registered assets/pools. The source-release manifest does not supply or certify those inputs. Runtime identities can differ from historical result identities because the adapter manifests have been reduced to retained files; numerical source bytes are unchanged.

No upstream license was declared in the pinned snapshot. Attribution and upstream terms remain with its authors. See `THIRD_PARTY_NOTICES.txt`.
