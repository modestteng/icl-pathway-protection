"""Five manuscript four-layer trajectories, uncapped adjacent label-K path selection.

The exp040/047 numerical kernels remain frozen. This adapter supplies only new
seed/provenance, exact monitor-argmax timing, scheduling and complete state I/O.
Checkpoints at a 100k boundary contain the already-advanced eval key; forks and
resumes never advance that key again at the same boundary.
"""
import argparse
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
SEEDS = json.loads((HERE / "protocol.json").read_text())["seeds"]
HORIZON, CONTINUATION, EVERY = 20_000_000, 5_600_000, 100_000
STOP = False


def stopped(signum, frame):
    global STOP
    STOP = True


def configure(args):
    global D, B, R, E, np, jax, OUT, ROOT, JOB, IDENTITY
    ROOT = args.root.resolve()
    output = ROOT / "results"
    assets = (args.assets or Path(os.environ.get("ICL040_ASSETS", str(ROOT / "assets")))).resolve()
    data = (args.data or output / "data").resolve()
    os.environ.update(ICL040_ASSETS=str(assets), ICL040_OUTPUTS=str(output), ICL040_ARCH="depth")
    old = HERE.parent / "exp_047_cross_layer_label_k"
    sys.path.insert(0, str(old))
    spec = importlib.util.spec_from_file_location("_exp054_frozen047", old / "driver.py")
    D = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = D
    spec.loader.exec_module(D)
    D.boot()
    B, R, E, np, jax = D.B, D.R, D.E, D.np, D.jax
    B.DATA = data
    OUT = output / "depth"
    OUT.mkdir(parents=True, exist_ok=True)
    D.WORK, D.OUTPUTS, D.OUT, D.ASSETS = HERE, output, OUT, assets
    D.SEEDS = {s: dict(v) for s, v in SEEDS.items()}
    D.identity = identity
    D.R.code_identity = identity
    B.code_identity = identity
    D.require_validation = require_validation
    D.verify_inputs = verify_inputs
    D.select_operator("P25_uncapped")
    IDENTITY = identity()
    JOB = output / "jobs" / (args.stage + ("_" + args.seed if args.seed else ""))
    JOB.mkdir(parents=True, exist_ok=True)
    D.JOB, D.STARTED = JOB, time.monotonic()
    protocol = D.read(HERE / "protocol.json")
    assert protocol["architecture"] == dict(depth=4, d_model=128, num_heads=8)
    assert float(protocol["rho"]) == .25
    assert int(protocol.get("source_horizon", HORIZON)) == HORIZON
    assert int(protocol["continuation_sequences"]) == CONTINUATION
    actual = protocol["seeds"]
    if isinstance(actual, list):
        actual = {row["id"]: row for row in actual}
    assert set(actual) == set(SEEDS)
    for sid, seed in SEEDS.items():
        for key in ("init", "train"):
            assert actual[sid][key] == seed[key]
    return protocol


def identity():
    return D.sha(HERE / "code_manifest.json")


def code_checks():
    for name, digest in D.read(HERE / "code_manifest.json").items():
        assert D.sha(HERE / name) == digest, "Changed adapter: " + name
    for parent, names in [
        (HERE.parent / "exp_040_scale_generalization_v2", ("common.py", "science.py", "engine.py", "mechanism.py", "path_set.py", "projection.py", "mechanism_checks.py", "set_checks.py")),
        (HERE.parent / "exp_047_cross_layer_label_k", ("driver.py", "cross_layer.py")),
    ]:
        manifest = D.read(parent / "code_manifest.json")
        for name in names:
            assert D.sha(parent / name) == manifest[name], "Changed frozen dependency: " + name


def verify_inputs(require_gpu=True):
    code_checks()
    # Released settings/input-spec bytes match git HEAD and the retained live
    # environment, but their historical embedded manifest was not regenerated.
    # Bind these exact owning configuration bytes to this new run explicitly.
    for name, expected in D.read(HERE / "frozen_runtime_config_sha256.json").items():
        assert D.sha(HERE.parent / "exp_040_scale_generalization_v2" / name) == expected, "Changed frozen runtime configuration: " + name
    manifest_path = ROOT / "input_manifest.json"
    manifest = D.read(manifest_path)
    checked = {}
    for group, base in (("assets", D.ASSETS), ("data", B.DATA)):
        for name, row in manifest.get(group, {}).items():
            path = Path(name)
            assert not path.is_absolute() and ".." not in path.parts
            digest = row["sha256"] if isinstance(row, dict) else row
            target = base / path
            assert D.sha(target) == digest, "Changed input: " + str(target)
            checked[group + "/" + name] = digest
    # Also accept a flat files mapping with explicit absolute paths, or paths
    # prefixed assets/ and data/. No credential files are part of this manifest.
    for name, row in manifest.get("files", {}).items():
        digest = row["sha256"] if isinstance(row, dict) else row
        value = row.get("path", name) if isinstance(row, dict) else name
        path = Path(value)
        if not path.is_absolute():
            if path.parts[0] == "assets":
                path = D.ASSETS.joinpath(*path.parts[1:])
            elif path.parts[0] == "data":
                path = B.DATA.joinpath(*path.parts[1:])
            else:
                path = ROOT / path
        assert D.sha(path) == digest, "Changed input: " + str(path)
        checked[str(name)] = digest
    assert checked, "input_manifest.json must register all assets and fixed pools"
    required_assets = ["config.json", "data/omniglot_features_reordered.h5", "data/omniglot_features_norotate.h5"]
    required_assets += ["source/" + row["upstream_path"] for row in D.read(HERE.parents[1] / "reproducibility/upstream_dependency.json")["files"]]
    for name in required_assets:
        expected = manifest.get("assets", {}).get(name)
        if expected is None:
            # A flat manifest is valid only when it explicitly registers the
            # required absolute file or assets-relative key.
            matches = [row for key, row in manifest.get("files", {}).items()
                       if str(key) in ("assets/" + name, str(D.ASSETS / name))
                       or (isinstance(row, dict) and row.get("path") == str(D.ASSETS / name))]
            assert len(matches) == 1, "Required asset absent from manifest: " + name
    for name in ("mechanism_discovery.h5", "mechanism_confirm.h5", "protection.h5", "monitor.h5", "confirm.h5", "complete.json"):
        assert name in manifest.get("data", {}) or any(
            str(key) in ("data/" + name, str(B.DATA / name)) or (isinstance(row, dict) and row.get("path") == str(B.DATA / name))
            for key, row in manifest.get("files", {}).items()), "Required pool absent from manifest: " + name
    for row in D.read(HERE.parents[1] / "reproducibility/upstream_dependency.json")["files"]:
        assert D.sha(D.ASSETS / "source" / row["upstream_path"]) == row["sha256"]
    versions = dict(jax=jax.__version__, jaxlib_distribution=B.distribution_version("jaxlib"),
                    equinox=D.eqx.__version__, optax=B.optax.__version__, numpy=np.__version__, h5py=B.h5py.__version__)
    for name, expected in B.SPEC["pinned_versions"].items():
        actual = __import__("jaxlib").__version__ if name == "jaxlib" else versions[name]
        assert actual == expected, (name, actual, expected)
    assert not jax.config.jax_enable_x64 and jax.config.jax_default_matmul_precision is None
    assert not require_gpu or jax.default_backend() == "gpu", "CUDA required"
    assert B.ARCHITECTURE == dict(depth=4, d_model=128, num_heads=8) and B.BS == 32
    for module in (B.main, B.main_utils, B.models, B.samplers, B.opto):
        assert Path(module.__file__).resolve().parent == (D.ASSETS / "source").resolve()
    record = dict(passed=True, inputs=checked, input_manifest_sha256=D.sha(manifest_path),
                  versions=versions, backend=jax.default_backend(), architecture=B.ARCHITECTURE,
                  code_manifest_sha256=identity(), seeds=SEEDS)
    target = D.OUTPUTS / "provenance.json"
    if target.exists():
        assert D.read(target) == record, "Runtime provenance changed"
    else:
        D.write(target, record)
    return record


def require_validation():
    record = D.read(OUT / "validation.json")
    assert record["passed"] and record["code_manifest_sha256"] == identity()


def checkpoint(sid, arm, stage):
    folder = "normal" if arm == "A" else arm
    return OUT / sid / folder / "checkpoints" / f"{int(stage):011d}.eqx"


def lock(sid):
    value = D.read(OUT / sid / "timing.json")
    assert value["code_manifest_sha256"] == identity() and value["selection_locked"]
    assert value["end_stage"] - value["start_stage"] == CONTINUATION
    return value


def load_peak(sid):
    timing = lock(sid)
    path = checkpoint(sid, "A", timing["start_stage"])
    assert D.sha(path) == timing["start_checkpoint_sha256"]
    state = B.load_checkpoint(path, SEEDS[sid])
    assert int(state["iter"]) == timing["start_stage"]
    return state, timing


def window(sid):
    end = lock(sid)["end_stage"]
    return list(range(end - 1_000_000, end + 1, EVERY))


def evaluate(path, state, data, ids, sid, arm, role):
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            assert str(old["code_manifest_sha256"]) == identity()
            assert int(old["stage"]) == int(state["iter"])
        return
    row = R.evaluate(state["model"], data, ids)
    row.update(stage=np.asarray(state["iter"]), seed_id=np.asarray(sid), arm=np.asarray(arm),
               code_manifest_sha256=np.asarray(identity()), role=np.asarray(role))
    B.save_npz(path, row)


def validate():
    verify_inputs()
    seed = SEEDS[next(iter(SEEDS))]
    base = B.initial_checkpoint(seed)
    protect, _ = D.dataset("protection")
    import mechanism_checks
    import set_checks
    def factory(width, depth):
        opts = copy.deepcopy(B.load_opts(seed))
        opts.d_model, opts.depth = width, depth
        return B.main_utils.get_model_from_opts(opts, (512,))
    D.progress(phase="validate_single_paths")
    singles = mechanism_checks.run_checks(factory, protect["icl"][0][0], protect["icl"][1][0], architectures=((128, 4),), include_gradient=True)
    assert singles["passed"], "Single-path numerical/structural validation failed"
    D.write(OUT / "validation/single_paths.json", singles)
    D.progress(phase="validate_path_sets")
    sets = set_checks.run_set_checks(base["model"], protect["icl"][0][0], protect["icl"][1][0], B.SETTINGS)
    assert sets["passed"], "Path-set numerical/structural validation failed"
    D.write(OUT / "validation/path_sets.json", sets)
    checks = {}
    for arm, paths in (("A", ()), ("P25_uncapped", (0, 9, 18))):
        D.progress(phase="validate_serial_queue_resume", arm=arm)
        checks[arm] = D.queue_check(base, seed, arm, paths, "exp054_" + arm)
    value = dict(passed=True, single_paths_passed=singles["passed"], path_sets_passed=sets["passed"],
                 queue_checks=checks, code_manifest_sha256=identity())
    D.write(OUT / "validation.json", value)
    return value


def training(sid, arm, end, paths=(), source=False, resume=False):
    """Queue the frozen numerical updates, retaining post-evaluation states."""
    require_validation()
    D.select_operator(arm)
    directory = OUT / sid / ("normal" if arm == "A" else arm)
    directory.mkdir(parents=True, exist_ok=True)
    pointer = directory / ("source_resume.json" if source else "resume.json")
    seed = SEEDS[sid]
    if source:
        start, state, last_eval = 0, B.initial_checkpoint(seed), -1
        fork_sha = None
    else:
        state, timing = load_peak(sid)
        start, last_eval = timing["start_stage"], timing["start_stage"]
        fork_sha = timing["start_checkpoint_sha256"]
        if arm == "A":
            state = B.load_checkpoint(checkpoint(sid, "A", HORIZON), seed)
            last_eval = HORIZON
    stats = dict(steps=0, correction_norm_ratio_sum=0., update_norm_ratio_sum=0., conflicts=0., near_zero_steps=0, max_relative_violation=0.)
    if pointer.exists():
        assert resume, "Existing state requires explicitly inspected --resume"
        receipt = D.read(pointer)
        assert receipt["code_manifest_sha256"] == identity() and receipt["paths"] == list(paths)
        assert receipt["end_stage"] == end and receipt["fork_checkpoint_sha256"] == fork_sha
        path = D.OUTPUTS / receipt["checkpoint_path"]
        assert D.sha(path) == receipt["checkpoint_sha256"]
        state = B.load_checkpoint(path, seed)
        last_eval, stats = receipt["last_evaluated_iter"], receipt["stats"]
    curve_file = directory / "curve.json"
    curve = D.read(curve_file) if curve_file.exists() else dict(points=[])
    sample, (_, optimizer) = B.make_sampler(seed), B.template(seed)
    monitor, mids = D.dataset("monitor", reported_only=True, on_gpu=True)
    if source:
        monitor, mids = {"icl": monitor["icl"]}, {"icl": mids["icl"]}
    confirm, cids = (None, None) if source else D.dataset("confirm", reported_only=True, on_gpu=True)
    protect, _ = D.dataset("protection")
    cache = [] if arm == "A" else R.cache_protection(protect)
    last_time, last_stage = time.monotonic(), int(state["iter"])
    def save():
        path = checkpoint(sid, arm, state["iter"])
        digest = B.save_checkpoint(path, state)
        D.write(pointer, dict(seed=seed, arm=arm, paths=list(paths), iter=int(state["iter"]),
            start_stage=start, end_stage=end, last_evaluated_iter=last_eval, stats=stats,
            fork_checkpoint_sha256=fork_sha, checkpoint_path=B.rel(path), checkpoint_sha256=digest,
            eval_key_already_advanced_at_checkpoint=(int(state["iter"]) == last_eval), code_manifest_sha256=identity()))
        return path, digest
    # Peak/source boundary has already advanced eval_model_seed. Initial
    # protected start readouts deliberately do not split it for a second time.
    if not source and arm != "A" and int(state["iter"]) == start:
        evaluate(directory / "start_confirm" / f"{arm}_{start:011d}.npz", state, confirm, cids, sid, arm, "start_after_lock")
    while True:
        stage = int(state["iter"])
        if stage % EVERY == 0 and last_eval != stage:
            state["seeds"]["eval_model_seed"], _ = jax.random.split(state["seeds"]["eval_model_seed"])
            if source:
                path = directory / "monitor" / f"A_{stage:011d}.npz"
                evaluate(path, state, monitor, mids, sid, arm, "source_monitor_only")
                with np.load(path, allow_pickle=False) as row:
                    count = int(np.asarray(row["icl/acc"], np.int64).sum())
                    ce = float(np.asarray(row["icl/loss"], np.float64).mean())
                curve["points"] = sorted([p for p in curve["points"] if p["stage"] != stage]
                    + [dict(stage=stage, correct_count=count, n_queries=5000, accuracy=count / 5000, mean_ce=ce)], key=lambda p: p["stage"])
            elif stage in window(sid):
                evaluate(directory / "confirm" / f"{arm}_{stage:011d}.npz", state, confirm, cids, sid, arm, "terminal_confirm_after_lock")
            assert int(np.asarray(state["opt_state"][0].count)) == stage // 32
            assert bool(np.asarray(R.finite_candidate(state["model"], state["opt_state"])))
            last_eval = stage
            path, digest = save()
            if source:
                curve["points"][-1].update(checkpoint_sha256=digest)
                D.write(curve_file, curve)
            elapsed = max(time.monotonic() - last_time, 1e-12)
            D.progress(phase="source" if source else "arm", seed=sid, arm=arm, stage=stage, end=end,
                       sequences_per_wall_second=(stage - last_stage) / elapsed, stats=stats)
            last_time, last_stage = time.monotonic(), stage
        if STOP or (D.OUTPUTS / "STOP").exists():
            save()
            D.write(directory / "stopped.json", dict(stage=stage, checkpoint_retained=True, code_manifest_sha256=identity()))
            raise SystemExit(3)
        if stage >= end:
            assert stage == end
            break
        boundary = min(end, (stage // EVERY + 1) * EVERY)
        count = min(B.SETTINGS["queue_chunk"], (boundary - stage) // 32)
        candidate, status, _ = E.queue_steps(state, count, "A" if arm == "A" else "P", 1. if arm == "A" else .25, paths, sample, optimizer, cache, start)
        stats["steps"] += count
        stats["correction_norm_ratio_sum"] += float(status[:, 4].astype(np.float64).sum())
        stats["update_norm_ratio_sum"] += float(status[:, 3].astype(np.float64).sum())
        stats["conflicts"] += float(status[:, 5].astype(np.float64).sum())
        stats["near_zero_steps"] += int((status[:, 3] < .01).sum())
        stats["max_relative_violation"] = max(stats["max_relative_violation"], float(status[:, 2].max()))
        state = candidate
    final, digest = save()
    return dict(passed=True, seed=seed, arm=arm, start_stage=start, end_stage=end,
        final_checkpoint_path=B.rel(final), final_checkpoint_sha256=digest, fork_checkpoint_sha256=fork_sha,
        paths=list(paths), rho=1. if arm == "A" else .25, update_statistics_sum=stats,
        code_manifest_sha256=identity(), t0_eval_key_advanced_again=False)


def source(sid, resume):
    result = training(sid, "A", HORIZON, source=True, resume=resume)
    directory = OUT / sid / "normal"
    curve = D.read(directory / "curve.json")
    points = curve["points"]
    assert [p["stage"] for p in points] == list(range(0, HORIZON + 1, EVERY))
    chosen = min(points, key=lambda p: (-p["correct_count"], p["stage"]))
    start = chosen["stage"]
    value = dict(selection_locked=True, seed=SEEDS[sid], start_stage=start,
        end_stage=start + CONTINUATION, start_checkpoint_sha256=D.sha(checkpoint(sid, "A", start)),
        selected_correct_count=chosen["correct_count"], search_start=0, search_end=HORIZON,
        metric="full_label_icl_correct_count", n_queries=5000, full_labels=12800,
        tie_rule="earliest_stage", smoothing=False, confirm_used_for_selection=False,
        code_manifest_sha256=identity(), curve_sha256=D.sha(directory / "curve.json"))
    path = OUT / sid / "timing.json"
    if path.exists():
        assert D.read(path) == value
    else:
        D.write(path, value)
    D.write(directory / "complete.json", result)
    # Score the untouched confirm pool only after the peak lock. Keep all
    # checkpoint hashes/monitor arrays, but retain only the selected fork,
    # complete latest source state and A terminal-window states on disk.
    confirm, ids = D.dataset("confirm", reported_only=True, on_gpu=True)
    for stage in sorted(set([start, *window(sid)])):
        if stage > HORIZON:
            continue
        state = B.load_checkpoint(checkpoint(sid, "A", stage), SEEDS[sid])
        folder = "confirm" if stage in window(sid) else "start_confirm"
        evaluate(OUT / sid / "A" / folder / f"A_{stage:011d}.npz", state, confirm, ids, sid, "A", "adopted_A_after_lock")
    trim_source_checkpoints(sid)
    return dict(source=result, timing=value)


def trim_source_checkpoints(sid):
    """Retention only: delete this new run's unneeded complete source states."""
    timing = lock(sid)
    keep = {timing["start_stage"], HORIZON, *window(sid)}
    if checkpoint(sid, "A", timing["end_stage"]).exists():
        keep.add(timing["end_stage"])
    directory = OUT / sid / "normal/checkpoints"
    removed = []
    for path in directory.glob("*.eqx"):
        stage = int(path.stem)
        if stage not in keep:
            removed.append(dict(stage=stage, checkpoint_sha256=D.sha(path), bytes=path.stat().st_size))
            path.unlink()
    record_path = OUT / sid / "normal/checkpoint_retention.json"
    old = D.read(record_path) if record_path.exists() else {"removed": []}
    prior = {v["stage"]: v for v in old["removed"]}
    prior.update({v["stage"]: v for v in removed})
    D.write(record_path, dict(removed=[prior[k] for k in sorted(prior)],
        retained_stages=sorted(int(p.stem) for p in directory.glob("*.eqx")),
        rule="Retain peak, full latest/final and A terminal window; hashes and all monitor arrays retained",
        only_new_exp054_checkpoints=True, code_manifest_sha256=identity()))


def discover(sid):
    require_validation()
    base, timing = load_peak(sid)
    discovery, dids = D.dataset("mechanism_discovery")
    import mechanism
    candidates = tuple(mechanism.candidate_path_ids(base["model"]))
    directory = OUT / sid / "discovery"
    scores, records = {}, {}
    for candidate in candidates:
        path = directory / "single" / f"{candidate}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                record = {name: saved[name] for name in saved.files}
        else:
            record, audit = D.mechanism_evaluate(base["model"], discovery, dids, (candidate,))
            assert R.structure_passed(audit), "Singleton structural failure"
            B.save_npz(path, record)
        assert list(record["path_set"]) == [candidate]
        scores[candidate] = float(np.mean([R.gain(record, task, "path").mean() for task in B.SETTINGS["protection_tasks"]]))
        records[candidate] = record
        D.progress(phase="discovery_single", seed=sid, candidate=candidate, candidate_count=len(candidates))
    ranking = sorted(scores, key=lambda p: (-scores[p], p))
    coverage, prefixes = [], []
    for k in range(1, len(ranking) + 1):
        paths = tuple(sorted(ranking[:k]))
        if k == 1:
            record = records[ranking[0]]
        else:
            record, audit = D.mechanism_evaluate(base["model"], discovery, dids, paths)
            assert R.structure_passed(audit), "Prefix structural failure"
        B.save_npz(directory / f"prefix{k}.npz", record)
        cov, _ = R.coverage(record)
        coverage.append(cov)
        prefixes.append(paths)
        D.progress(phase="uncapped_prefix", seed=sid, k=k, candidate_count=len(candidates), coverage=cov)
    k, reason = D.PS.choose_prefix(coverage, .5)
    paths = prefixes[k - 1]
    selected = dict(seed=SEEDS[sid], selected=list(paths), candidates=list(candidates), candidate_count=len(candidates),
        ranking=ranking, scores={str(p): scores[p] for p in ranking}, prefix_coverage=coverage, k=k,
        k_max=len(candidates), artificial_cap=None, coverage_target=.5, selection_reason=reason,
        selection_data="mechanism_discovery", mechanism_confirm_used_for_selection=False,
        confirm_used_for_selection=False, fork_checkpoint_sha256=timing["start_checkpoint_sha256"],
        start_stage=timing["start_stage"], rho=.25, code_manifest_sha256=identity())
    D.write(directory / "selection.json", selected)
    confirmation, cids = D.dataset("mechanism_confirm")
    record, audit = D.mechanism_evaluate(base["model"], confirmation, cids, paths)
    assert R.structure_passed(audit)
    B.save_npz(directory / "mechanism_confirm.npz", record)
    summary = D.mechanism_summary(record, cids, audit, paths)
    assert summary["structure_passed"]
    D.write(directory / "decision.json", dict(confirmation=summary,
        qualified=bool(summary["passed"] and summary["coverage"] is not None and summary["coverage"] >= .5),
        train_despite_failed_functional_gate=True, selected=list(paths), code_manifest_sha256=identity()))
    check = D.queue_check(base, SEEDS[sid], "P25_uncapped", paths, sid + "_selected")
    D.write(directory / "queue_validation.json", dict(passed=True, queue_check=check,
        code_manifest_sha256=identity(), fork_checkpoint_sha256=timing["start_checkpoint_sha256"]))
    D.write(directory / "complete.json", dict(passed=True, selection=selected, code_manifest_sha256=identity()))
    return selected


def arm(sid, resume):
    timing = lock(sid)
    selected = D.read(OUT / sid / "discovery/selection.json")
    assert selected["code_manifest_sha256"] == identity()
    assert D.read(OUT / sid / "discovery/queue_validation.json")["passed"]
    assert selected["fork_checkpoint_sha256"] == timing["start_checkpoint_sha256"]
    # Reuse the full unprotected trajectory; extend it only beyond its 20M
    # frozen peak-search horizon. This never changes the locked peak.
    a_dir = OUT / sid / "A"
    if (a_dir / "complete.json").exists():
        result = D.read(a_dir / "complete.json")
        assert result["passed"] and result["code_manifest_sha256"] == identity()
    elif timing["end_stage"] > HORIZON:
        result = training(sid, "A", timing["end_stage"], resume=resume)
    else:
        result = dict(passed=True, adopted_source_checkpoints=True, end_stage=timing["end_stage"], code_manifest_sha256=identity())
    confirm, ids = D.dataset("confirm", reported_only=True, on_gpu=True)
    for stage in sorted(set([timing["start_stage"], *window(sid)])):
        state = B.load_checkpoint(checkpoint(sid, "A", stage), SEEDS[sid])
        folder = "confirm" if stage in window(sid) else "start_confirm"
        evaluate(a_dir / folder / f"A_{stage:011d}.npz", state, confirm, ids, sid, "A", "adopted_A_after_lock")
    D.write(a_dir / "complete.json", result)
    trim_source_checkpoints(sid)
    p_result = training(sid, "P25_uncapped", timing["end_stage"], tuple(selected["selected"]), resume=resume)
    D.write(OUT / sid / "P25_uncapped/complete.json", p_result)
    return dict(A=result, P25_uncapped=p_result)


def summarize():
    result = dict(schema="exp054_five_fresh_seed_uncapped_summary_v1", seeds={}, rho=.25,
        statistical_unit="training_seed", source_horizon=HORIZON, continuation_sequences=CONTINUATION,
        code_manifest_sha256=identity(), all_seeds_retained=True)
    for sid in SEEDS:
        rows = {}
        for arm_name in ("A", "P25_uncapped"):
            assert D.read(OUT / sid / arm_name / "complete.json")["passed"]
            values, hashes = {}, {}
            for stage in window(sid):
                path = OUT / sid / arm_name / "confirm" / f"{arm_name}_{stage:011d}.npz"
                hashes[str(path.relative_to(D.OUTPUTS))] = D.sha(path)
                with np.load(path, allow_pickle=False) as saved:
                    assert str(saved["code_manifest_sha256"]) == identity()
                    for task in B.TASKS:
                        for metric in ("acc", "loss"):
                            key = task + "/" + metric
                            values.setdefault(key, []).append(float(np.asarray(saved[key], np.float64).mean()))
            rows[arm_name] = dict(status="complete", window_points=11, means={k: float(np.mean(v)) for k, v in values.items()}, confirm_files_sha256=hashes)
        rows["paired_differences"] = {key: rows["P25_uncapped"]["means"][key] - rows["A"]["means"][key] for key in values}
        rows["selection"] = D.read(OUT / sid / "discovery/selection.json")
        rows["timing"] = lock(sid)
        result["seeds"][sid] = rows
    result["across_seed_mean_differences"] = {key: float(np.mean([v["paired_differences"][key] for v in result["seeds"].values()])) for key in values}
    D.write(D.OUTPUTS / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--stage", choices=("validate", "source", "discover", "arm", "summarize"), required=True)
    parser.add_argument("--seed", choices=tuple(SEEDS))
    parser.add_argument("--resume", action="store_true", help="Explicitly resume an inspected incomplete/failing stage")
    args = parser.parse_args()
    if args.stage in ("source", "discover", "arm") and not args.seed:
        parser.error("This stage requires --seed")
    configure(args)
    handle = (JOB / "run.lock").open("a")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (JOB / "complete.json").exists():
        assert D.read(JOB / "complete.json")["code_manifest_sha256"] == identity()
        D.progress(phase="already_complete", stage=args.stage, seed=args.seed)
        return
    if (JOB / "failure.json").exists() and not args.resume:
        raise RuntimeError("Preserve existing failure; inspect before explicit --resume")
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        verify_inputs()
        if args.stage == "validate":
            value = validate()
        else:
            require_validation()
            if args.stage == "source":
                value = source(args.seed, args.resume)
            elif args.stage == "discover":
                value = discover(args.seed)
            elif args.stage == "arm":
                value = arm(args.seed, args.resume)
            else:
                value = summarize()
        D.write(JOB / "complete.json", dict(passed=True, result=value, code_manifest_sha256=identity(), utc=D.now()))
        D.progress(phase="completed", stage=args.stage, seed=args.seed)
    except BaseException as error:
        D.write(JOB / "failure.json", dict(error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), code_manifest_sha256=identity(), utc=D.now()))
        raise


if __name__ == "__main__":
    main()
