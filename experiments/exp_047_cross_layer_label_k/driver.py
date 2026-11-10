"""Shared label-K utilities for the current manuscript's four-layer adapters.

Reuses exp040's model, samplers, Adam, projector, queues and statistics.
The manuscript adapters inject their registered seed IDs and run settings.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

WORK = Path(__file__).resolve().parent
LEGACY = WORK.parent / "exp_040_scale_generalization_v2"
ROOT = WORK.parents[1]
SEEDS = {}
ARMS = ("P25_old", "P25_cross")
os.environ.setdefault("ICL040_ASSETS", str(WORK / "assets"))
os.environ.setdefault("ICL040_OUTPUTS", str(WORK / "outputs"))
os.environ["ICL040_ARCH"] = "depth"
OUTPUTS = Path(os.environ["ICL040_OUTPUTS"]).resolve()
OUT = OUTPUTS / "depth"
ASSETS = Path(os.environ["ICL040_ASSETS"]).resolve()
JOB = None
STOP = False
STARTED = time.monotonic()
BOOTED = False


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def identity():
    return sha(WORK / "code_manifest.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def progress(**values):
    record = dict(utc=now(), pid=os.getpid(), elapsed_seconds=time.monotonic() - STARTED, **values)
    if JOB is not None:
        write(JOB / "progress.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def boot():
    global BOOTED, B, R, E, C, PS, np, jax, jnp, eqx, PATH_VALUE
    if BOOTED:
        return
    sys.path.insert(0, str(LEGACY))
    import common as B
    import science as R
    import engine as E
    import cross_layer as C
    import path_set as PS
    np, jax, jnp, eqx = B.np, B.jax, B.jnp, B.eqx
    # The reused update-audit function looks this name up in science's globals.
    # Bind its receipt to the actual new driver/kernel protocol, not exp040's
    # manifest. Scientific kernels and optimizer operations are unchanged.
    R.code_identity = identity
    # Reuse one value-only executable across trial batches; do not embed each
    # batch as a new closure constant. Match functional_objective's precision.
    def path_value(model, x, y, keys, paths):
        with jax.default_matmul_precision("float32"):
            return R.PS.set_loss(model, x, y, keys, paths)
    PATH_VALUE = eqx.filter_jit(path_value)
    BOOTED = True


def verify_inputs(require_gpu=True):
    """Own rebuilt identities; never call exp040's unavailable historical gate."""
    boot()
    code = read(WORK / "code_manifest.json")
    for name, expected in code.items():
        if sha(WORK / name) != expected:
            raise RuntimeError("exp047 sealed code changed: " + name)
    protocol = read(WORK / "protocol.json")
    expected_seeds = {sid: dict(init=row["init"], train=row["train"],
        start_stage=row["start"], end_stage=row["end"]) for sid, row in SEEDS.items()}
    if protocol["seeds"] != expected_seeds or protocol["rho"] != .25 or protocol["arms"] != ["A", *ARMS]:
        raise RuntimeError("Protocol differs from the user-authorized two-seed scope")
    legacy_manifest = read(LEGACY / "code_manifest.json")
    used = ("common.py", "science.py", "engine.py", "datasets.py", "mechanism.py",
            "path_set.py", "projection.py", "set_checks.py", "mechanism_checks.py")
    for name in used:
        if sha(LEGACY / name) != legacy_manifest[name]:
            raise RuntimeError("exp040 dependency changed: " + name)
    manifest_path = ASSETS / "assets_manifest.json"
    manifest = read(manifest_path)
    if manifest.get("schema") != "exp047_rebuilt_assets_v1" or manifest.get("historical_feature_hash_equality") is not False:
        raise ValueError("Rebuilt asset manifest must explicitly disavow historical feature identity")
    files = manifest["files"]
    required = {"config.json", "data/omniglot_features_reordered.h5", "data/omniglot_features_norotate.h5"}
    upstream = read(ROOT / "reproducibility/upstream_dependency.json")
    required.update("source/" + row["upstream_path"] for row in upstream["files"])
    if not required.issubset(files):
        raise ValueError("Missing required asset identities: " + str(sorted(required - set(files))))
    digests = {}
    for name, row in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe asset path")
        expected = row["sha256"] if isinstance(row, dict) else row
        digests[name] = sha(ASSETS / relative)
        if digests[name] != expected:
            raise RuntimeError("Rebuilt asset changed: " + name)
    for row in upstream["files"]:
        if digests["source/" + row["upstream_path"]] != row["sha256"]:
            raise RuntimeError("Pinned upstream source mismatch: " + row["upstream_path"])
    versions = dict(jax=jax.__version__, jaxlib=__import__("jaxlib").__version__,
        jaxlib_distribution=B.distribution_version("jaxlib"), equinox=eqx.__version__,
        optax=B.optax.__version__, numpy=np.__version__, h5py=B.h5py.__version__)
    for name, expected in B.SPEC["pinned_versions"].items():
        if versions[name] != expected:
            raise RuntimeError(f"Pinned version mismatch {name}: {versions[name]} != {expected}")
    if require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("CUDA required; CPU fallback is forbidden")
    if jax.config.jax_enable_x64 or jax.config.jax_default_matmul_precision is not None:
        raise RuntimeError("Original float32/default global matmul semantics required")
    for module in [B.main, B.main_utils, B.models, B.samplers, B.opto]:
        if Path(module.__file__).resolve().parent != ASSETS / "source":
            raise RuntimeError("Imported unintended upstream source: " + str(module.__file__))
    if B.BS != 32 or B.SETTINGS["path_set"]["k_max"] != 4 or B.SETTINGS["path_set"]["coverage_target"] != .5:
        raise RuntimeError("Original batch/source-selection rules changed")
    record = dict(schema="exp047_provenance_v1", historical_feature_hash_equality=False,
        asset_manifest_sha256=sha(manifest_path), inputs=digests, versions=versions,
        code_manifest_sha256=identity(), legacy_code_manifest_sha256=sha(LEGACY / "code_manifest.json"),
        architecture=B.ARCHITECTURE, seeds=SEEDS, rho=.25,
        devices=[str(d) for d in jax.devices()], backend=jax.default_backend())
    old = OUTPUTS / "provenance.json"
    if old.exists() and read(old) != record:
        raise RuntimeError("Runtime provenance changed; preserve prior attempt before resuming")
    write(old, record)
    return record


def select_operator(arm):
    boot()
    R.PS = C if arm == "P25_cross" else PS


def seed_spec(seed_id):
    source = SEEDS[seed_id]
    return {name: source[name] for name in ("id", "init", "train")}


def window(seed_id):
    end = SEEDS[seed_id]["end"]
    return list(range(end - 1000000, end + 1, 100000))


def dataset(name, reported_only=False, on_gpu=False):
    data, ids = B.load_dataset(name)
    if reported_only:
        data = {task: data[task] for task in B.TASKS}
        ids = {task: ids[task] for task in B.TASKS}
    return (R.stage_on_gpu(data) if on_gpu else data), ids


def evaluate(path, model, data, ids, **metadata):
    if path.exists():
        return
    record = R.evaluate(model, data, ids)
    record.update({name: np.asarray(value) for name, value in metadata.items()})
    B.save_npz(path, record)


def data_stage():
    verify_inputs(require_gpu=True)
    import datasets as D
    D.verify_inputs = verify_inputs
    D.code_identity = identity
    D.SETTINGS = dict(B.SETTINGS, version="exp047_cross_layer_label_k_rebuilt_inputs")
    result = D.prepare()
    write(OUTPUTS / "data_usage.json", dict(discovery="mechanism_discovery only",
        mechanism_gate="mechanism_confirm only", protection="protection only",
        update_direction="monitor[:128] ICL only, never used for source selection",
        terminal_evaluation="confirm only; never used for source selection",
        historical_feature_hash_equality=False, source_seeds=B.SETTINGS["data_sets"]))
    return result


def queue_check(base, seed, arm, paths, label):
    select_operator(arm)
    protect, _ = dataset("protection")
    sample = B.make_sampler(seed)
    _, optimizer = B.template(seed)
    cache = R.cache_protection(protect)
    kind = "A" if arm == "A" else "P"
    rho = 1.0 if kind == "A" else .25
    start = int(base["iter"])
    begun = time.monotonic()
    serial, ss = E.serial_steps(base, 64, kind, rho, paths, sample, optimizer, protect, start)
    serial_seconds = time.monotonic() - begun
    begun = time.monotonic()
    queued, qs, _ = E.queue_steps(base, 64, kind, rho, paths, sample, optimizer, cache, start)
    queued_seconds = time.monotonic() - begun
    mid, _, _ = E.queue_steps(base, 32, kind, rho, paths, sample, optimizer, cache, start)
    path = OUT / "validation" / f"{label}_32steps.eqx"
    B.save_checkpoint(path, mid)
    restored = B.load_checkpoint(path, seed)
    resumed, _, _ = E.queue_steps(restored, 32, kind, rho, paths, sample, optimizer, cache, start)
    same = dict(model_exact=B.exact(serial["model"], queued["model"]),
        adam_exact=B.exact(serial["opt_state"], queued["opt_state"]),
        prng_exact=B.exact(serial["seeds"], queued["seeds"]),
        status_exact=bool(np.array_equal(ss, qs)), reload_exact=B.exact(mid, restored),
        resumed_model_exact=B.exact(serial["model"], resumed["model"]),
        resumed_adam_exact=B.exact(serial["opt_state"], resumed["opt_state"]),
        resumed_prng_exact=B.exact(serial["seeds"], resumed["seeds"]))
    if not all(same.values()):
        raise AssertionError(("Queue/serial/resume mismatch", label, same))
    return dict(passed=True, **same, paths=list(paths), rho=rho,
                serial_seconds=serial_seconds, queued_seconds=queued_seconds)


def validate_stage():
    verify_inputs()
    import test_cross_layer
    seed = seed_spec(next(iter(SEEDS)))
    base = B.initial_checkpoint(seed)
    protect, _ = dataset("protection")
    progress(phase="validate_cross_layer_structure")
    structural = test_cross_layer.run_checks(base["model"], protect["icl"][0][0],
                                            protect["icl"][1][0], B.SETTINGS)
    write(OUT / "validation" / "cross_layer_structure.json", structural)
    checks = {}
    for arm, paths in [("A", ()), ("P25_old", (0, 9)),
                       ("P25_cross", ((0, 2, 0), (1, 3, 1)))]:
        progress(phase="validate_queue", arm=arm)
        checks[arm] = queue_check(base, seed, arm, paths, arm)
    # Exercise metadata adapters and the actual loss/gradient/CE pipeline now,
    # before committing hours to rebuilding training states. The initial model
    # is only an integration fixture and need not pass scientific qualification.
    integration = integration_smoke(base, seed)
    result = dict(passed=True, queue_checks=checks,
        structure_sha256=sha(OUT / "validation" / "cross_layer_structure.json"),
        integration_smoke=integration,
        code_manifest_sha256=identity(), seed=seed, scientific_fixture_updates_committed=False)
    write(OUT / "validation.json", result)
    return result


def require_validation():
    receipt = read(OUT / "validation.json")
    if not receipt["passed"] or receipt["code_manifest_sha256"] != identity():
        raise RuntimeError("Current code lacks successful CUDA validation")


def checkpoint_path(seed_id, arm, stage):
    folder = "normal" if arm == "A" else arm
    return OUT / seed_id / folder / "checkpoints" / f"{stage:011d}.eqx"


def load_fork(seed_id):
    path = checkpoint_path(seed_id, "A", SEEDS[seed_id]["start"])
    normal = read(OUT / seed_id / "normal" / "complete.json")
    if normal["code_manifest_sha256"] != identity() or normal["fork_checkpoint_sha256"] != sha(path):
        raise RuntimeError("Rebuilt fork identity changed")
    return B.load_checkpoint(path, seed_spec(seed_id)), sha(path)


def run_training(seed_id, arm="A", paths=()):
    require_validation()
    select_operator(arm)
    seed = seed_spec(seed_id)
    cfg = SEEDS[seed_id]
    directory = OUT / seed_id / ("normal" if arm == "A" else arm)
    directory.mkdir(parents=True, exist_ok=True)
    pointer = directory / "resume.json"
    start, end = (0 if arm == "A" else cfg["start"]), cfg["end"]
    fork_digest = None
    if arm == "A":
        base = B.initial_checkpoint(seed)
    else:
        base, fork_digest = load_fork(seed_id)
    stats = dict(steps=0, correction_norm_ratio_sum=0., update_norm_ratio_sum=0.,
                 conflicts=0., near_zero_steps=0, max_relative_violation=0.)
    if pointer.exists():
        resume = read(pointer)
        if resume["code_manifest_sha256"] != identity() or resume["paths"] != json.loads(json.dumps(list(paths))):
            raise RuntimeError("Resume code/source-set changed")
        if resume["fork_checkpoint_sha256"] != fork_digest:
            raise RuntimeError("Resume fork changed")
        cp = OUTPUTS / resume["checkpoint_path"]
        if sha(cp) != resume["checkpoint_sha256"]:
            raise RuntimeError("Resume checkpoint changed")
        ck = B.load_checkpoint(cp, seed)
        last_eval, stats = resume["last_evaluated_iter"], resume["stats"]
    else:
        ck, last_eval = base, -1
    monitor, mids = dataset("monitor", reported_only=True, on_gpu=True)
    monitor_all, _ = dataset("monitor")
    confirm, cids = dataset("confirm", reported_only=True, on_gpu=True)
    protect, _ = dataset("protection")
    cache = R.cache_protection(protect) if arm != "A" else []
    sample, (_, optimizer) = B.make_sampler(seed), B.template(seed)
    kind = "A" if arm == "A" else "P"
    rho = 1.0 if arm == "A" else .25
    last_progress_time, last_progress_stage = time.monotonic(), int(ck["iter"])

    def save():
        stage = int(ck["iter"])
        cp = checkpoint_path(seed_id, arm, stage)
        digest = B.save_checkpoint(cp, ck)
        write(pointer, dict(iter=stage, seed=seed, arm=arm, paths=list(paths), stats=stats,
            last_evaluated_iter=last_eval, start_stage=start, end_stage=end,
            fork_checkpoint_sha256=fork_digest, checkpoint_path=B.rel(cp), checkpoint_sha256=digest,
            code_manifest_sha256=identity()))
        return cp, digest

    while True:
        stage = int(ck["iter"])
        if stage % B.EVERY == 0 and last_eval != stage:
            # Preserve the original stage-0/every-100k eval-key advances, even
            # where fixed timing means no actual monitor evaluation is needed.
            ck["seeds"]["eval_model_seed"], _ = jax.random.split(ck["seeds"]["eval_model_seed"])
            meta = dict(stage=stage, seed_id=seed_id, arm=arm,
                        code_manifest_sha256=identity(), historical_feature_hash_equality=False)
            if stage == cfg["start"] or stage in window(seed_id):
                evaluate(directory / "monitor" / f"{arm}_{stage:011d}.npz", ck["model"], monitor, mids, **meta)
            if stage in window(seed_id):
                target = OUT / seed_id / "A" if arm == "A" else directory
                evaluate(target / "confirm" / f"{arm}_{stage:011d}.npz", ck["model"], confirm, cids, **meta)
            if int(np.asarray(ck["opt_state"][0].count)) != stage // B.BS:
                raise RuntimeError("Adam count and sequence count disagree")
            if not bool(np.asarray(R.finite_candidate(ck["model"], ck["opt_state"]))):
                raise RuntimeError("Nonfinite committed state")
            last_eval = stage
            save()
            elapsed = time.monotonic() - last_progress_time
            progress(phase="normal" if arm == "A" else "arm", seed=seed_id, arm=arm,
                     stage=stage, start=start, end=end, paths=list(paths), rho=rho,
                     sequences_per_wall_second=(stage - last_progress_stage) / max(elapsed, 1e-12), stats=stats)
            last_progress_time, last_progress_stage = time.monotonic(), stage
        if STOP or (OUTPUTS / "STOP").exists():
            save()
            write(directory / "stopped.json", dict(stage=int(ck["iter"]), checkpoint_retained=True, utc=now()))
            raise SystemExit(3)
        if stage == end:
            break
        boundary = min(end, (stage // B.EVERY + 1) * B.EVERY)
        count = min(B.SETTINGS["queue_chunk"], (boundary - stage) // B.BS)
        offsets = tuple(B.SETTINGS["diagnostic_update_offsets"]) if arm != "A" else ()
        offset = (stage - start) // B.BS
        future = [v - offset for v in offsets if v > offset]
        if future:
            count = min(count, min(future))
        if offset in offsets:
            count = 1
        candidate, status, audit = E.queue_steps(ck, count, kind, rho, paths, sample, optimizer, cache, start, offsets)
        if audit is not None:
            i, old, new, raw, gradients, px, py = audit
            R.audit_update(directory, seed, arm, i, old, new, raw, gradients, px, py, monitor_all, paths, kind)
            if offset == 0:
                _, dk, _, mk = E.split_keys(ck["seeds"]["train_data_seed"], ck["seeds"]["train_model_seed"])
                batch = sample(dk)
                author = B.sync(B.main.train_step(model=ck["model"], fwd_fn=R.FWD, optimizer=optimizer,
                    opt_state=ck["opt_state"], microbs=B.BS, weight_decay=0.,
                    x=batch["examples"], y=batch["labels"], key=mk))
                if not B.exact(candidate["opt_state"], author[2]):
                    raise AssertionError("First protected-step Adam state differs from author")
        stats["steps"] += count
        stats["correction_norm_ratio_sum"] += float(status[:, 4].astype(np.float64).sum())
        stats["update_norm_ratio_sum"] += float(status[:, 3].astype(np.float64).sum())
        stats["conflicts"] += float(status[:, 5].astype(np.float64).sum())
        stats["near_zero_steps"] += int((status[:, 3] < .01).sum())
        stats["max_relative_violation"] = max(stats["max_relative_violation"], float(status[:, 2].max()))
        ck = candidate
    final, final_digest = save()
    result = dict(passed=True, seed=seed, arm=arm, start_stage=start, end_stage=end,
        final_checkpoint_path=B.rel(final), final_checkpoint_sha256=final_digest,
        fork_checkpoint_sha256=(sha(checkpoint_path(seed_id, "A", cfg["start"])) if arm == "A" else fork_digest),
        paths=list(paths), rho=rho, update_statistics_sum=stats,
        code_manifest_sha256=identity(), historical_feature_hash_equality=False,
        window_stages=window(seed_id), elapsed_seconds=time.monotonic() - STARTED)
    write(directory / "complete.json", result)
    return result


def mechanism_evaluate(model, data, ids, paths):
    """Retain exp040 batched metrics/audits without its integer-only adapter."""
    record, max_audit = {}, {}
    bs = B.SETTINGS["mechanism_bs"]
    for task in B.SETTINGS["protection_tasks"]:
        x, y = data[task]
        collect = {}
        for lo in range(0, len(x), bs):
            hi = min(lo + bs, len(x))
            metrics, _, _ = R.mechanism_batch(model, jnp.asarray(R._pad(x[lo:hi], bs)),
                jnp.asarray(R._pad(y[lo:hi], bs)), tuple(paths), False, False)
            for name, values in metrics.items():
                values = np.asarray(values)[:hi - lo]
                if not np.isfinite(values).all():
                    raise RuntimeError("Nonfinite mechanism metric")
                collect.setdefault(name, []).append(values)
                if name.startswith(("audit/", "reverse_audit/")):
                    max_audit[name] = max(max_audit.get(name, 0.), float(values.max()))
        for name, values in collect.items():
            record[task + "/" + name] = np.concatenate(values)
        record[task + "/identities"] = ids[task]
    record["path_set"] = np.asarray(paths, np.int32)
    return record, max_audit


def path_name(path):
    return "_".join(map(str, path)) if isinstance(path, tuple) else str(path)


def mechanism_summary(record, ids, audit, paths):
    """Original exp040 tests/thresholds; only path metadata supports triples.

    The original summary ends with ``[int(p) for p in path_ids]``. Its statistical
    functions are reused here without encoding cross-layer paths as integers.
    """
    tasks = tuple(B.SETTINGS["protection_tasks"])
    checks, floors = {}, {}
    for task in tasks:
        classes = ids[task][:, -1, 0]
        donor, key_gain, value_gain = (R.gain(record, task, stem) for stem in ("donor", "path", "v_path"))
        contrasts = [
            ("donor_ce", donor, B.SETTINGS["mechanism_donor_ce_min"]),
            ("path_ce", key_gain, B.SETTINGS["mechanism_path_ce_min"]),
            ("retention", key_gain - B.SETTINGS["mechanism_retention_min"] * donor, 0.),
            ("k_vs_v", key_gain - value_gain, B.SETTINGS["mechanism_k_vs_v_min"]),
            ("pair_margin", np.asarray(record[task + "/path/donor_margin"], np.float64)
                - np.asarray(record[task + "/native/donor_margin"], np.float64),
                B.SETTINGS["mechanism_pair_margin_min"]),
        ]
        for name, values, threshold in contrasts:
            checks[task + "/" + name] = R.cluster_test(values, classes, threshold)
        for field, floor in [("native/native_pair_acc", B.SETTINGS["mechanism_pair_acc_floor"]),
                             ("donor/donor_pair_acc", B.SETTINGS["mechanism_pair_acc_floor"]),
                             ("native/native_full_acc", B.SETTINGS["mechanism_full_acc_floor"])]:
            result = R.cluster_test(record[task + "/" + field], classes, floor)
            result["passed"] = bool(np.isfinite(result["se"]) and result["se"] > 1e-15
                                    and result["lower95"] > floor)
            floors[task + "/" + field] = result
        if not bool(np.all(record[task + "/pair_valid"])) or not np.all(
                record[task + "/native_target"] != record[task + "/donor_target"]):
            raise AssertionError("Invalid source/recipient pairing in confirmation")
    for name, p in zip(checks, R.holm([row["p"] for row in checks.values()])):
        checks[name].update(p_holm=p, passed=bool(p < B.SETTINGS["alpha"]
            and checks[name]["mean"] > checks[name]["threshold"]))
    structural = R.structure_passed(audit)
    for task in tasks:
        for prediction in ("full_pred", "pair_pred"):
            structural = structural and np.array_equal(record[task + "/self/" + prediction],
                                                        record[task + "/native/" + prediction])
    coverage, ratios = R.coverage(record)
    return dict(passed=bool(structural and all(row["passed"] for row in checks.values())
                            and all(row["passed"] for row in floors.values())),
        path_set=list(paths), path_set_description=R.PS.describe(paths), checks=checks,
        floors=floors, structure_passed=bool(structural), audit_max=audit,
        coverage=coverage, coverage_by_task=ratios, tasks=list(tasks))


def restore_paths(value, arm):
    return tuple(tuple(map(int, p)) for p in value) if arm == "P25_cross" else tuple(map(int, value))


def update_diagnostic(seed_id, arm, base, paths, directory, *, reference_checkpoint=None,
                      validation_only=False, trial_batch_index=0):
    """Trial updates on the exact same main/protection/independent ICL batches."""
    select_operator(arm)
    seed = seed_spec(seed_id)
    sample, (_, optimizer) = B.make_sampler(seed), B.template(seed)
    if not isinstance(trial_batch_index, int) or not 0 <= trial_batch_index < 8:
        raise ValueError("The bounded diagnostic admits exactly indices 0 through 7")
    # Advance only copied keys. Every trial starts from the SAME model and Adam
    # state at t0; no earlier trial's update or optimizer state is committed.
    copied_data_key = base["seeds"]["train_data_seed"]
    copied_model_key = base["seeds"]["train_model_seed"]
    for _ in range(trial_batch_index + 1):
        copied_data_key, dk, copied_model_key, mk = E.split_keys(copied_data_key, copied_model_key)
    main_batch = B.sync(sample(dk))
    protect, _ = dataset("protection")
    px, py = R.protect_batch(protect, trial_batch_index)
    xx, yy = px.reshape(-1, 3, 512), py.reshape(-1, 3)
    pk = jax.random.split(B.KEY, len(xx))
    path_loss, path_grad = R.functional_objective(base["model"], xx, yy, pk, paths)
    author = B.sync(B.main.train_step(model=base["model"], fwd_fn=R.FWD, optimizer=optimizer,
        opt_state=base["opt_state"], microbs=B.BS, weight_decay=0.,
        x=main_batch["examples"], y=main_batch["labels"], key=mk))
    raw = R.adam_update_from_state(author[2])
    raw_model = eqx.apply_updates(base["model"], raw)
    reconstruction_error = B.maxdiff(raw_model, author[1])
    if reconstruction_error > B.SETTINGS["numeric_model_adam_max_abs"]:
        raise AssertionError("Reconstructed raw Adam update differs from author")
    projected, diag = R.project_tree(raw, (path_grad,), .25)
    if not bool(np.asarray(diag["success"])) or float(np.asarray(diag["relative_constraint_violation"])) > B.SETTINGS["numeric_projection_relative_violation_max"]:
        raise AssertionError("Trial protection projection failed")
    corrected = eqx.apply_updates(base["model"], projected)
    monitor, _ = dataset("monitor")
    mx, my = [jnp.asarray(v[:128]) for v in monitor["icl"]]
    nk = jax.random.split(B.KEY, len(mx))
    # main.compute_loss is already value-and-grad; task_objective preserves its
    # (loss, gradient) return. A second differentiation would treat a gradient
    # tree as part of the scalar output and is invalid.
    natural_loss, natural_grad = R.task_objective(base["model"], mx, my, nk)
    flat = lambda tree: R.flat(tree).astype(np.float64)
    ng, pg, ru, pu = map(flat, (natural_grad, path_grad, raw, projected))
    if not all(np.isfinite(v).all() for v in (ng, pg, ru, pu)):
        raise AssertionError("Nonfinite trial gradient or update")
    correction = pu - ru
    denominator = np.linalg.norm(ng) * np.linalg.norm(pg)
    losses = {}
    for label, model in [("before", base["model"]), ("raw", raw_model), ("corrected", corrected)]:
        losses[label] = dict(natural_icl=float(np.asarray(R.task_objective(model, mx, my, nk)[0])),
                             path=float(np.asarray(PATH_VALUE(model, xx, yy, pk, paths))))
    if not all(np.isfinite(value) for pair in losses.values() for value in pair.values()):
        raise AssertionError("Nonfinite actual trial cross-entropy")

    def batch_digest(examples, labels):
        arrays = [np.asarray(v) for v in (examples, labels)]
        h = hashlib.sha256()
        for value in arrays:
            h.update(json.dumps(dict(shape=value.shape, dtype=str(value.dtype)), sort_keys=True).encode())
            h.update(value.tobytes())
        return h.hexdigest()

    reference = (checkpoint_path(seed_id, "A", int(base["iter"]))
                 if reference_checkpoint is None else Path(reference_checkpoint))
    reference_digest = sha(reference)
    result = dict(seed=seed_spec(seed_id), arm=arm, paths=list(paths), rho=.25,
        stage=int(base["iter"]), trial_models_committed=False, selection_used_this_diagnostic=False,
        trial_batch_index=trial_batch_index, main_batches_advanced=trial_batch_index+1,
        main_data_prng_key=np.asarray(dk).tolist(), main_model_prng_key=np.asarray(mk).tolist(),
        same_t0_model_and_adam_for_every_trial=True,
        validation_only=validation_only,
        reference_checkpoint_role="validation_fixture" if validation_only else "rebuilt_t0_fork",
        reference_checkpoint_path=B.rel(reference), reference_checkpoint_sha256=reference_digest,
        fork_checkpoint_sha256=None if validation_only else reference_digest,
        main_batch_sha256=batch_digest(main_batch["examples"], main_batch["labels"]),
        protection_batch_sha256=batch_digest(px, py),
        monitor_batch_sha256=batch_digest(mx, my),
        monitor_rows=128, ordered_monitor_indices=list(range(128)),
        protection_task_order=list(B.SETTINGS["protection_tasks"]),
        protection_rows_per_task=32,
        ordered_protection_indices=((np.arange(32)+trial_batch_index*32)
            % B.SETTINGS["data_sets"]["protection"]["n"]).tolist(),
        diagnostic_data="monitor/icl rows 0:128",
        cos_natural_icl_path_gradient=(float(ng @ pg / denominator) if denominator else None),
        natural_icl_gradient_dot_correction=float(ng @ correction),
        natural_icl_gradient_norm=float(np.linalg.norm(ng)), path_gradient_norm=float(np.linalg.norm(pg)),
        path_gradient_dot_raw=float(pg @ ru), path_gradient_dot_corrected=float(pg @ pu),
        path_gradient_dot_correction=float(pg @ correction),
        projection_active=bool(np.asarray(diag["projected"])),
        correction_norm=float(np.linalg.norm(correction)),
        correction_nonzero=bool(np.linalg.norm(correction)>0),
        interpretation="Positive g_ICL dot correction predicts worse natural ICL than raw Adam; negative predicts better, locally.",
        predicted_natural_raw_change=float(ng @ ru), predicted_natural_corrected_change=float(ng @ pu),
        actual_natural_raw_change=losses["raw"]["natural_icl"] - losses["before"]["natural_icl"],
        actual_natural_corrected_change=losses["corrected"]["natural_icl"] - losses["before"]["natural_icl"],
        actual_natural_corrected_minus_raw=losses["corrected"]["natural_icl"] - losses["raw"]["natural_icl"],
        losses=losses, raw_adam_model_reconstruction_max_abs=reconstruction_error,
        code_manifest_sha256=identity())
    write(directory / "natural_icl_update_diagnostics" / f"trial_{trial_batch_index:02d}.json", result)
    if trial_batch_index == 0:
        write(directory / "natural_icl_update_diagnostic.json", result)
    return result


def update_diagnostic_series(seed_id, arm, base, paths, directory):
    """Eight fixed trial batches; descriptive only, never a source/arm gate."""
    rows = [update_diagnostic(seed_id, arm, base, paths, directory, trial_batch_index=index)
            for index in range(8)]
    active = [row for row in rows if row["correction_nonzero"]]
    summary = dict(seed=seed_spec(seed_id), arm=arm, trial_batch_indices=list(range(8)),
        n_trials=8, same_t0_model_and_adam=True, trial_updates_committed=False,
        selection_used_diagnostic=False, diagnostic_used_as_training_gate=False,
        projection_active_count=sum(row["projection_active"] for row in rows),
        nonzero_correction_count=len(active), zero_correction_count=8-len(active),
        predicted_improving_active_count=sum(row["natural_icl_gradient_dot_correction"] < 0 for row in active),
        predicted_worsening_active_count=sum(row["natural_icl_gradient_dot_correction"] > 0 for row in active),
        predicted_zero_active_count=sum(row["natural_icl_gradient_dot_correction"] == 0 for row in active),
        actual_improving_active_count=sum(row["actual_natural_corrected_minus_raw"] < 0 for row in active),
        actual_worsening_active_count=sum(row["actual_natural_corrected_minus_raw"] > 0 for row in active),
        actual_zero_active_count=sum(row["actual_natural_corrected_minus_raw"] == 0 for row in active),
        trials=[dict(trial_batch_index=row["trial_batch_index"],
            main_batch_sha256=row["main_batch_sha256"], protection_batch_sha256=row["protection_batch_sha256"],
            correction_norm=row["correction_norm"], projection_active=row["projection_active"],
            cos_natural_icl_path_gradient=row["cos_natural_icl_path_gradient"],
            natural_icl_gradient_dot_correction=row["natural_icl_gradient_dot_correction"],
            actual_natural_corrected_minus_raw=row["actual_natural_corrected_minus_raw"])
            for row in rows], code_manifest_sha256=identity())
    write(directory / "natural_icl_update_diagnostics" / "summary.json", summary)
    return summary


def integration_smoke(base, seed):
    """Small CUDA integration fixture; no behavioral qualification is required."""
    directory = OUT / "validation" / "integration_smoke"
    reference = directory / "initial_fixture.eqx"
    B.save_checkpoint(reference, base)
    data, ids = dataset("mechanism_confirm")
    # Fixed, bounded rows; distinct from protection and monitor calibration.
    # This is not discovery, target selection, or an end-point evaluation.
    rows = {task: (xy[0][:64], xy[1][:64]) for task, xy in data.items()}
    row_ids = {task: values[:64] for task, values in ids.items()}
    results = {}
    for arm, paths in [("P25_old", (0, 9)),
                       ("P25_cross", ((0, 2, 0), (1, 3, 1)))]:
        select_operator(arm)
        progress(phase="validate_integration_smoke", arm=arm, validation_only=True)
        record, audits = mechanism_evaluate(base["model"], rows, row_ids, paths)
        if not R.structure_passed(audits):
            raise AssertionError("Integration fixture violated structural invariants: " + arm)
        summary = mechanism_summary(record, row_ids, audits, paths)
        # Finite/schema execution is required. Random initial weights need not
        # satisfy effect sizes, accuracy floors, coverage, or qualification.
        json.dumps(summary, allow_nan=False)
        if summary["path_set"] != list(paths) or not isinstance(summary["passed"], bool):
            raise AssertionError("Mechanism metadata adapter schema mismatch")
        arm_directory = directory / arm
        B.save_npz(arm_directory / "mechanism_fixture.npz", record)
        write(arm_directory / "mechanism_summary.json", dict(validation_only=True,
            scientific_qualification_required=False, data_pool="mechanism_confirm",
            ordered_indices=list(range(64)), summary=summary))
        diagnostic = update_diagnostic(seed["id"], arm, base, paths, arm_directory,
            reference_checkpoint=reference, validation_only=True)
        json.dumps(diagnostic, allow_nan=False)
        if set(diagnostic["losses"]) != {"before", "raw", "corrected"} or diagnostic["trial_models_committed"]:
            raise AssertionError("Update diagnostic integration schema mismatch")
        results[arm] = dict(passed=True, validation_only=True, scientific_qualification_required=False,
            observed_fixture_qualification=summary["passed"], mechanism_rows_per_task=64,
            natural_icl_calibration_rows=diagnostic["monitor_rows"],
            mechanism_summary_sha256=sha(arm_directory / "mechanism_summary.json"),
            update_diagnostic_sha256=sha(arm_directory / "natural_icl_update_diagnostic.json"))
    write(directory / "complete.json", dict(passed=True, validation_only=True,
        trial_updates_committed=False, results=results, code_manifest_sha256=identity()))
    return results


def discover_stage(seed_id):
    require_validation()
    seed = seed_spec(seed_id)
    base, digest = load_fork(seed_id)
    discovery, dids = dataset("mechanism_discovery")
    confirmation, cids = dataset("mechanism_confirm")
    results = {}
    for arm in ARMS:
        select_operator(arm)
        directory = OUT / seed_id / "discovery" / arm
        directory.mkdir(parents=True, exist_ok=True)
        candidates = C.candidate_paths(base["model"]) if arm == "P25_cross" else tuple(range(24))
        scores, singleton_files = {}, {}
        for candidate in candidates:
            path = directory / "single" / (path_name(candidate) + ".npz")
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    record = {k: saved[k] for k in saved.files}
            else:
                record, audits = mechanism_evaluate(base["model"], discovery, dids, (candidate,))
                if not R.structure_passed(audits):
                    raise AssertionError("Discovery operator structural failure: " + str(candidate))
                B.save_npz(path, record)
            scores[candidate] = float(np.mean([R.gain(record, task, "path").mean()
                                              for task in B.SETTINGS["protection_tasks"]]))
            singleton_files[candidate] = path
            progress(phase="discovery_single", seed=seed_id, arm=arm, candidate=candidate, score=scores[candidate])
        ranking = sorted(candidates, key=lambda p: (-scores[p], p))
        coverages, prefixes = [], []
        for k in range(1, 5):
            paths = tuple(sorted(ranking[:k]))
            if k == 1:
                with np.load(singleton_files[ranking[0]], allow_pickle=False) as saved:
                    record = {name: saved[name] for name in saved.files}
            else:
                record, audits = mechanism_evaluate(base["model"], discovery, dids, paths)
                if not R.structure_passed(audits):
                    raise AssertionError("Discovery prefix structural failure")
                B.save_npz(directory / f"prefix{k}.npz", record)
            coverage, _ = R.coverage(record)
            coverages.append(coverage)
            prefixes.append(paths)
        k, reason = PS.choose_prefix(coverages, .5)
        selected = prefixes[k - 1]
        write(directory / "selection.json", dict(seed=seed, arm=arm, selected=list(selected),
            candidates=list(candidates), ranking=list(ranking), scores=[dict(path=p, score=scores[p]) for p in ranking],
            prefix_coverage=coverages, k=k, k_max=4, coverage_target=.5, selection_reason=reason,
            selection_data="mechanism_discovery", fork_checkpoint_sha256=digest,
            code_manifest_sha256=identity()))
        record, audits = mechanism_evaluate(base["model"], confirmation, cids, selected)
        B.save_npz(directory / "mechanism_confirm.npz", record)
        summary = mechanism_summary(record, cids, audits, selected)
        qualified = bool(summary["passed"] and summary["coverage"] is not None and summary["coverage"] >= .5)
        decision = dict(seed=seed, arm=arm, qualified=qualified, selected=list(selected),
            confirmation=summary, confirmation_data="mechanism_confirm", fork_checkpoint_sha256=digest,
            old_target_unqualified_training_allowed=(arm == "P25_old"),
            cross_unqualified_training_allowed=False, code_manifest_sha256=identity())
        write(directory / "decision.json", decision)
        if arm == "P25_cross" and not qualified:
            write(directory / "NOT_QUALIFIED.json", dict(decision="do_not_train", reason="independent_functional_gate_failed", **decision))
        update_diagnostic_series(seed_id, arm, base, selected, directory)
        results[arm] = decision
        progress(phase="source_qualification", seed=seed_id, arm=arm, qualified=qualified,
                 selected=list(selected), confirmation_coverage=summary["coverage"])
    return results


def arm_stage(seed_id, arm):
    decision = read(OUT / seed_id / "discovery" / arm / "decision.json")
    if decision["code_manifest_sha256"] != identity():
        raise RuntimeError("Source decision belongs to different code")
    if arm == "P25_cross" and not decision["qualified"]:
        progress(phase="NOT_QUALIFIED", seed=seed_id, arm=arm,
                 reason="independent functional validation failed; no protected continuation started")
        return dict(status="NOT_QUALIFIED", training_started=False)
    paths = restore_paths(decision["selected"], arm)
    select_operator(arm)
    base, _ = load_fork(seed_id)
    check = queue_check(base, seed_spec(seed_id), arm, paths, f"{seed_id}_{arm}")
    write(OUT / seed_id / arm / "fork_queue_validation.json", check)
    return run_training(seed_id, arm, paths)


def summarize_stage():
    import numpy as np
    result = dict(schema="exp047_two_seed_summary_v1", historical_feature_hash_equality=False,
        statistical_unit="two specifically selected training seeds; descriptive, no generalization claim",
        rho=.25, source_selection_cap=4, seeds={})
    for sid in SEEDS:
        seed_result = {}
        for arm in ("A", *ARMS):
            directory = OUT / sid / arm
            files = [directory / "confirm" / f"{arm}_{stage:011d}.npz" for stage in window(sid)]
            if not all(path.exists() for path in files):
                not_qualified = OUT / sid / "discovery" / arm / "NOT_QUALIFIED.json"
                seed_result[arm] = dict(status="NOT_QUALIFIED" if not_qualified.exists() else "incomplete")
                continue
            metrics = {}
            for path in files:
                with np.load(path, allow_pickle=False) as rows:
                    for task in ("icl", "iwl_copy_avail", "train_eval"):
                        for metric in ("acc", "loss"):
                            name = task + "/" + metric
                            metrics.setdefault(name, []).append(float(np.asarray(rows[name], np.float64).mean()))
            seed_result[arm] = dict(status="complete", window_points=11,
                means={name: float(np.mean(values)) for name, values in metrics.items()},
                confirm_files_sha256={str(path.relative_to(OUTPUTS)): sha(path) for path in files})
        if all(seed_result[a]["status"] == "complete" for a in ("A", "P25_old", "P25_cross")):
            seed_result["paired_differences"] = {
                f"{a}_minus_{b}": {name: seed_result[a]["means"][name] - seed_result[b]["means"][name]
                    for name in seed_result[a]["means"]}
                for a, b in [("P25_cross", "A"), ("P25_cross", "P25_old"), ("P25_old", "A")]}
        result["seeds"][sid] = seed_result
    write(OUTPUTS / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def signal_stop(signum, frame):
    global STOP
    STOP = True


def main():
    global JOB
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("data", "validate", "normal", "discover", "arm", "summarize"))
    parser.add_argument("--seed", choices=tuple(SEEDS))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--resume", action="store_true", help="Explicitly resume a previously inspected failed/stopped stage")
    args = parser.parse_args()
    if args.stage in ("normal", "discover", "arm") and args.seed is None:
        parser.error("This stage requires a registered --seed")
    if args.stage == "arm" and args.arm is None:
        parser.error("arm requires --arm P25_old|P25_cross")
    name = "_".join(v for v in (args.stage, args.seed, args.arm) if v)
    JOB = OUTPUTS / "jobs" / name
    JOB.mkdir(parents=True, exist_ok=True)
    if (JOB / "failure.json").exists() and not args.resume:
        raise RuntimeError("Prior failure retained. Diagnose before explicit --resume; automatic retry forbidden")
    if args.stage != "summarize" and (JOB / "complete.json").exists():
        if read(JOB / "complete.json")["code_manifest_sha256"] != identity():
            raise RuntimeError("Completed stage belongs to different sealed code")
        progress(phase="already_complete", stage=args.stage, seed=args.seed, arm=args.arm)
        return
    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)
    progress(phase="setup", stage=args.stage, seed=args.seed, arm=args.arm)
    exit_code = 1
    try:
        if args.stage == "summarize":
            result = summarize_stage()
        else:
            verify_inputs()
            if args.stage == "data":
                result = data_stage()
            elif args.stage == "validate":
                result = validate_stage()
            elif args.stage == "normal":
                result = run_training(args.seed)
            elif args.stage == "discover":
                result = discover_stage(args.seed)
            else:
                result = arm_stage(args.seed, args.arm)
        write(JOB / "complete.json", dict(passed=True, utc=now(), result=result,
            code_manifest_sha256=identity(), elapsed_seconds=time.monotonic() - STARTED))
        exit_code = 0
    except BaseException as error:
        exit_code = error.code if isinstance(error, SystemExit) and isinstance(error.code, int) else 1
        if exit_code != 3:
            write(JOB / "failure.json", dict(error_type=type(error).__name__, error=str(error),
                traceback=traceback.format_exc(), utc=now(), automatic_retry=False,
                code_manifest_sha256=identity()))
        raise
    finally:
        write(JOB / "exit.json", dict(exit_code=exit_code, utc=now(),
            elapsed_seconds=time.monotonic() - STARTED, pid=os.getpid()))


if __name__ == "__main__":
    raise SystemExit("Use the manuscript adapters in exp_054 or exp_055; this module supplies shared utilities.")
