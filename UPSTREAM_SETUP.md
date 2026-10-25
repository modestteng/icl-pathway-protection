# Upstream source dependency

The model, sampler, intervention and Omniglot preparation implementation is an external dependency from [aadityasingh/icl-dynamics](https://github.com/aadityasingh/icl-dynamics). Those seven source files and their frozen duplicates are omitted from this public repository. Saved-result checks do not need them.

The dependency is pinned to upstream commit `85b895c720844e795734b9391b32d2619065f9a1`. Each file is byte-identical to the frozen source used by the experiments; the SHA-256 values are in `reproducibility/upstream_dependency.json`. The historical `source_manifest.json` retains its original snapshot identity.

To restore the dependencies locally from the original repository, run:

```bash
python3 tools/prepare_upstream.py --fetch
python3 tools/verify_package.py --require-upstream
```

Alternatively, obtain an upstream checkout separately and provide its location:

```bash
python3 tools/prepare_upstream.py --source-dir ../icl-dynamics
python3 tools/verify_package.py --require-upstream
```

The preparation command checks all seven hashes before writing, refuses to replace different existing files, and does not execute the downloaded source or install an environment. It restores the root imports and the matching frozen source locations expected by the experiment scripts. These generated local files are ignored by Git and excluded from the public release manifest.

Training and model evaluation also require the recorded scientific environment and external data, features and checkpoints. Restoring source alone does not constitute a successful experimental rerun. See `reproducibility/ENVIRONMENT_GUIDE.txt`, `DATA_SOURCES.md`, and the experiment protocols.

The upstream repository had no declared license when checked for this release. Its source remains under the upstream authors' terms; this release does not assign it a license or claim redistribution permission.
