"""Frozen feature extraction / forward evaluation; no training code is called."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
import urllib.request

from io_utils import ROOT, OUT, CFG, sha, write_json, verify_package


def read_idx(path):
    import numpy as np
    with gzip.open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        nd = magic & 255
        assert magic >> 8 == 8 and nd in (1, 3)
        shape = struct.unpack('>' + 'I' * nd, f.read(4 * nd))
        a = np.frombuffer(f.read(), np.uint8).reshape(shape).copy()
    return a


def prepare():
    import numpy as np
    import h5py
    from PIL import Image, ImageDraw
    OUT.mkdir(exist_ok=True)
    raw = ROOT / 'raw'
    raw.mkdir(exist_ok=True)
    downloads, download_start = {}, time.monotonic()
    for name, expected in CFG['downloads'].items():
        p = raw / name
        url = ('https://huggingface.co/datasets/' + CFG['hf_repository'] + '/resolve/' +
               CFG['hf_revision'] + '/' + name)
        if not p.exists():
            tmp = p.with_suffix(p.suffix + '.partial')
            if tmp.exists():
                raise RuntimeError('partial download present; inspect before retry')
            print('download', name, flush=True)
            with urllib.request.urlopen(url, timeout=120) as r, tmp.open('wb') as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            tmp.replace(p)
        if sha(p) != expected:
            raise RuntimeError('raw download SHA mismatch: ' + name)
        downloads[name] = dict(url=url, sha256=expected, bytes=p.stat().st_size)
    download_seconds = time.monotonic() - download_start
    start = time.monotonic()
    arrays = {}
    for split, n, count in [('train', 112800, 2400), ('test', 18800, 400)]:
        images = read_idx(raw / ('emnist-balanced-' + split + '-images-idx3-ubyte.gz'))
        labels = read_idx(raw / ('emnist-balanced-' + split + '-labels-idx1-ubyte.gz'))
        assert images.shape == (n, 28, 28) and labels.shape == (n,)
        assert np.array_equal(np.bincount(labels, minlength=47), np.full(47, count))
        arrays[split] = images, labels
    selected, used, skips = [], set(), dict(train=0, test=0)
    for split, seed in [('train', CFG['support_seed']), ('test', CFG['query_seed'])]:
        images, labels = arrays[split]
        for c in range(47):
            rng = np.random.default_rng(np.random.SeedSequence([seed, c]))
            picked = 0
            for i in rng.permutation(np.flatnonzero(labels == c)):
                im = images[int(i)]
                digest = hashlib.sha256(im.tobytes()).hexdigest()
                if digest in used:
                    skips[split] += 1
                    continue
                used.add(digest)
                selected.append((im.copy(), c, int(split == 'test'), int(i), digest))
                picked += 1
                if picked == 50:
                    break
            assert picked == 50, ('not enough distinct images', split, c)
    assert len(selected) == 4700 and len(used) == 4700
    pool_images = np.stack([x[0] for x in selected])
    classes = np.asarray([x[1] for x in selected], np.int32)
    splits = np.asarray([x[2] for x in selected], np.uint8)
    indices = np.asarray([x[3] for x in selected], np.int64)
    pixel_hash = np.asarray([x[4].encode() for x in selected], 'S64')
    episode_ids, targets = [], []
    for c in range(47):
        rng = np.random.default_rng(np.random.SeedSequence([CFG['episode_seed'], c]))
        target = rng.permutation(np.repeat([0, 1], 25))
        match_position = rng.permutation(np.repeat([0, 1], 25))
        for j in range(50):
            other = int(rng.integers(46))
            other += int(other >= c)
            match_id = c * 50 + int(rng.integers(50))
            distractor_id = other * 50 + int(rng.integers(50))
            context = [distractor_id, distractor_id]
            context[int(match_position[j])] = match_id
            query_id = 2350 + c * 50 + j
            ys = [1 - int(target[j]), 1 - int(target[j])]
            ys[int(match_position[j])] = int(target[j])
            episode_ids.append(context + [query_id])
            targets.append(ys + [int(target[j])])
    episode_ids = np.asarray(episode_ids, np.int32)
    targets = np.asarray(targets, np.int32)
    query_classes = classes[episode_ids[:, -1]]
    out = OUT / 'selected_images.h5'
    assert not out.exists()
    with h5py.File(out, 'w') as f:
        for name, a in dict(images=pool_images, classes=classes, official_split=splits,
                            original_index=indices, pixel_sha256=pixel_hash).items():
            f.create_dataset(name, data=a, compression='gzip')
        f.attrs['format_adapter'] = CFG['image_adapter']
    np.savez_compressed(OUT / 'episodes.npz', image_ids=episode_ids, labels=targets,
                        query_classes=query_classes)
    # A data inspection image, produced before model outputs are available.
    canvas = Image.new('RGB', (7 * 92, 7 * 105), 'white')
    draw = ImageDraw.Draw(canvas)
    for c in range(47):
        image = Image.fromarray(255 - pool_images[c * 50].T, mode='L').resize((84, 84))
        x, y = (c % 7) * 92, (c // 7) * 105
        canvas.paste(image, (x, y + 17))
        draw.text((x + 2, y + 2), 'class ' + str(c), fill='black')
    canvas.save(OUT / 'sample_preview.png')
    write_json(OUT / 'data_manifest.json', dict(
        raw_files=downloads, source='NIST EMNIST Balanced via fixed HF mirror revision',
        counts=dict(classes=47, images=4700, support=2350, queries=2350),
        raw_duplicate_skips=skips, unique_selected_pixel_hashes=4700,
        support_query_pixel_overlap=0, download_seconds=download_seconds,
        preparation_seconds=time.monotonic() - start,
        selected_images_sha256=sha(out), episodes_sha256=sha(OUT / 'episodes.npz'),
        seeds={k:CFG[k] for k in ['support_seed', 'query_seed', 'episode_seed']}))
    print('data_complete', json.dumps(dict(images=4700, queries=2350, skips=skips)), flush=True)


def encode():
    import numpy as np
    import h5py
    import torch
    import torchvision
    from PIL import Image
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from torchvision.models import resnet18, ResNet18_Weights
    start = time.monotonic()
    assert torch.__version__.split('+')[0] == '2.2.2' and torchvision.__version__.split('+')[0] == '0.17.2'
    assert torch.cuda.is_available(), 'Windows CUDA required'
    torch.set_num_threads(4)
    old = json.loads(Path(CFG['original_encoder_provenance']).read_text())
    weights = Path(torch.hub.get_dir()) / 'checkpoints/resnet18-f37072fd.pth'
    assert weights.exists(), 'reuse original cached weights; no substitute download'
    assert sha(weights) == old['weights_sha256']
    assert torch.backends.cuda.matmul.allow_tf32 == old['cuda_matmul_allow_tf32']
    assert torch.backends.cudnn.allow_tf32 == old['cudnn_allow_tf32']
    transform = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with h5py.File(OUT / 'selected_images.h5', 'r') as f:
        images = f['images'][:]

    class Images(Dataset):
        def __len__(self):
            return len(images)

        def __getitem__(self, i):
            return transform(Image.fromarray(255 - images[i].T, mode='L'))

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model = model.to('cuda').eval()

    def parameter_digest():
        h = hashlib.sha256()
        for key, value in model.state_dict().items():
            h.update(key.encode())
            h.update(value.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    before = parameter_digest()
    loader = DataLoader(Images(), batch_size=64, shuffle=False, num_workers=2,
                        pin_memory=True, persistent_workers=False)
    torch.cuda.reset_peak_memory_stats()
    result = np.empty((4700, 512), np.float32)
    events, offset = [], 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to('cuda', non_blocking=True)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            z = model(batch)
            e1.record()
            assert z.device.type == 'cuda' and z.dtype == torch.float32
            a = z.cpu().numpy()
            assert a.shape == (len(batch), 512) and np.isfinite(a).all()
            result[offset:offset + len(a)] = a
            offset += len(a)
            events.append((e0, e1))
            if offset % 640 == 0 or offset == len(images):
                print('encoded', offset, flush=True)
                write_json(OUT / 'encoding_progress.json', dict(done=offset, total=4700))
        audit = torch.stack([Images()[i] for i in range(4)]).to('cuda')
        # Keep the production convolution shape while using only four distinct images.
        # Native batch4 selects different FP32/TF32 convolution arithmetic on this GPU.
        audit = torch.cat([audit, audit[-1:].repeat(60, 1, 1, 1)])
        repeat = model(audit).cpu().numpy()[:4]
    torch.cuda.synchronize()
    max_error = float(np.max(np.abs(repeat - result[:4])))
    print('four_image_same_shape_repeat_max_abs', max_error, flush=True)
    assert max_error <= 1e-4
    after = parameter_digest()
    assert before == after, 'encoder parameters or BN state changed'
    fpath = OUT / 'features.h5'
    assert not fpath.exists()
    with h5py.File(fpath, 'w') as f:
        f.create_dataset('features', data=result)
        f.attrs['encoder'] = CFG['encoder']
        f.attrs['weights_sha256'] = sha(weights)
        f.attrs['selected_images_sha256'] = sha(OUT / 'selected_images.h5')
        f.attrs['format_adapter'] = CFG['image_adapter']
    write_json(OUT / 'encoding_receipt.json', dict(
        status='complete', shape=list(result.shape), dtype=str(result.dtype),
        device=torch.cuda.get_device_name(0), weights_sha256=sha(weights),
        original_weights_sha256=old['weights_sha256'], parameters_before=before,
        parameters_after=after, four_image_repeat_max_abs=max_error,
        validation_logical_batch_size=4, validation_computational_batch_size=64,
        torch=torch.__version__, torchvision=torchvision.__version__,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        original_cuda_matmul_allow_tf32=old.get('cuda_matmul_allow_tf32'),
        original_cudnn_allow_tf32=old.get('cudnn_allow_tf32'),
        gpu_forward_seconds=sum(a.elapsed_time(b) for a,b in events) / 1000,
        encoding_stage_seconds=time.monotonic() - start,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        features_sha256=sha(fpath)))
    print('encoding_complete', flush=True)


def evaluate(arm):
    import numpy as np
    import h5py
    import importlib.metadata as md
    assets = Path(CFG['assets'])
    for name, expected in CFG['source_file_sha256'].items():
        assert sha(assets / 'source' / name) == expected, name
    assert sha(assets / 'config.json') == CFG['config_sha256']
    path = Path(CFG['models'][arm]['path'])
    assert sha(path) == CFG['models'][arm]['sha256']
    sys.path.insert(0, str(assets / 'source'))
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import main
    import main_utils
    import opto
    from functools import partial
    for name, expected in [('jax', '0.4.26'), ('jaxlib', '0.4.26+cuda12.cudnn89'),
                           ('equinox', '0.11.4'), ('optax', '0.2.2')]:
        assert md.version(name) == expected, name
    assert jax.default_backend() == 'gpu'
    assert not jax.config.jax_enable_x64 and jax.config.jax_default_matmul_precision is None
    start = time.monotonic()
    opts = main_utils.get_opts_from_json_file(assets / 'config.json')
    template = main_utils.get_model_from_opts(opts, (512,))
    key = jax.random.PRNGKey(0)
    optimizer = main_utils.get_optimizer_from_opts(opts)
    ck = dict(iter=-1, model=template,
              opt_state=optimizer.init(eqx.filter(template, eqx.is_array)),
              seeds=dict(eval_model_seed=key, train_data_seed=key, train_model_seed=key))
    ck = eqx.tree_deserialise_leaves(path, ck)
    assert int(ck['iter']) == CFG['stage']
    assert int(np.asarray(ck['opt_state'][0].count)) == CFG['stage'] // 32
    model = ck['model']
    del ck, optimizer, template
    for layer in jax.tree_util.tree_leaves(model, is_leaf=lambda v:isinstance(v,eqx.nn.Dropout)):
        if isinstance(layer,eqx.nn.Dropout):
            assert layer.p == 0, 'unexpected stochastic dropout'
    fwd = opto.make_fn_from_opts(opts)

    @eqx.filter_jit
    def forward(m, x, y):
        logits = jax.vmap(partial(fwd, model=m))(
            x=x, y=y, key=jax.random.split(key, len(x)))['out'][:, -1, :]
        return logits, main.ce(logits, y[:, -1])

    with h5py.File(OUT / 'features.h5', 'r') as f:
        features = f['features'][:]
    d = np.load(OUT / 'episodes.npz')
    ids, y = d['image_ids'], d['labels']
    x = features[ids]
    raw0, ce0 = forward(model, jnp.asarray(x[:32]), jnp.asarray(y[:32]))
    native = main.eval_step(model=model, fwd_fn=fwd, x=jnp.asarray(x[:32]),
                            y=jnp.asarray(y[:32]), key=key)
    raw0, ce0 = np.asarray(raw0), np.asarray(ce0)
    assert np.array_equal(np.argmax(raw0, 1) == y[:32, -1], np.asarray(native['acc']))
    assert np.array_equal(np.argmax(raw0[:, :2], 1) == y[:32, -1], np.asarray(native['in_context_acc']))
    native_error = float(np.max(np.abs(ce0 - np.asarray(native['loss']))))
    assert native_error <= 2e-5
    outpath = OUT / (arm + '_logits.h5')
    assert not outpath.exists()
    stored = {name:[] for name in ['full_pred', 'ce', 'pair_pred', 'pair_ce', 'context_mass']}
    bs, n = CFG['eval_batch_size'], len(y)
    with h5py.File(outpath, 'w') as f:
        zds = f.create_dataset('logits', (n, 12800), 'float32', chunks=(32, 12800), compression='lzf')
        for lo in range(0, n, bs):
            hi = min(n, lo + bs)
            xx, yy = x[lo:hi], y[lo:hi]
            if len(xx) < bs:
                xx = np.concatenate([xx, np.repeat(xx[-1:], bs-len(xx), axis=0)])
                yy = np.concatenate([yy, np.repeat(yy[-1:], bs-len(yy), axis=0)])
            logits, ce = forward(model, jnp.asarray(xx), jnp.asarray(yy))
            logits, ce = np.asarray(logits)[:hi-lo], np.asarray(ce)[:hi-lo]
            assert np.isfinite(logits).all() and np.isfinite(ce).all()
            zds[lo:hi] = logits
            pair = logits[:, :2].astype(np.float64)
            pmax = pair.max(1)
            fullmax = logits.max(1).astype(np.float64)
            logz = fullmax + np.log(np.exp(logits.astype(np.float64)-fullmax[:,None]).sum(1))
            pairz = pmax + np.log(np.exp(pair-pmax[:,None]).sum(1))
            stored['full_pred'].extend(logits.argmax(1))
            stored['ce'].extend(ce)
            stored['pair_pred'].extend(pair.argmax(1))
            stored['pair_ce'].extend(pairz-pair[np.arange(hi-lo), y[lo:hi,-1]])
            stored['context_mass'].extend(np.exp(pairz-logz))
            print(arm, 'evaluated', hi, '/', n, flush=True)
            write_json(OUT / (arm + '_progress.json'), dict(done=hi, total=n))
    arrays = {k:np.asarray(v) for k,v in stored.items()}
    arrays.update(target=y[:,-1], query_classes=d['query_classes'], image_ids=ids)
    np.savez_compressed(OUT / (arm + '_rows.npz'), **arrays)
    np.savez_compressed(OUT / (arm + '_audit_logits.npz'), logits=raw0, target=y[:32,-1])
    assert sha(path) == CFG['models'][arm]['sha256']
    write_json(OUT / (arm + '_evaluation_receipt.json'), dict(
        status='complete', arm=arm, stage=CFG['stage'], checkpoint_sha256=sha(path),
        device=[str(v) for v in jax.devices()], versions={k:md.version(k) for k in ['jax','jaxlib','equinox','optax']},
        rows=n, full_label_count=12800, native_audit_rows=32,
        native_ce_max_abs=native_error, evaluation_seconds=time.monotonic()-start,
        episodes_sha256=sha(OUT / 'episodes.npz'), features_sha256=sha(OUT / 'features.h5'),
        logits_sha256=sha(outpath), rows_sha256=sha(OUT / (arm + '_rows.npz'))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'encode', 'evaluate'])
    p.add_argument('--arm', choices=['A', 'P25'])
    a = p.parse_args()
    verify_package()
    if a.stage == 'prepare':
        prepare()
    elif a.stage == 'encode':
        encode()
    else:
        assert a.arm
        evaluate(a.arm)
