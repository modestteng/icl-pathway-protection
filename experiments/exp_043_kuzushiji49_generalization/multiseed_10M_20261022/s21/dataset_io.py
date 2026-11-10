"""Raw dataset readers and fixed image-format adapters; no model imports."""
import csv
import tarfile
import numpy as np
from io_utils import ROOT, CFG


def read_source(split):
    raw = ROOT / 'raw'
    if CFG['dataset_kind'] == 'k49':
        with np.load(raw / ('k49-' + split + '-imgs.npz'), allow_pickle=False) as f:
            images = f['arr_0'].copy()
        with np.load(raw / ('k49-' + split + '-labels.npz'), allow_pickle=False) as f:
            labels = f['arr_0'].copy()
    else:
        assert CFG['dataset_kind'] == 'cifar100'
        with tarfile.open(raw / 'cifar-100-binary.tar.gz', 'r:gz') as t:
            data = t.extractfile('cifar-100-binary/' + split + '.bin').read()
        a = np.frombuffer(data, np.uint8).reshape(-1, 3074)
        labels = a[:, 1].copy()  # Fine labels; coarse labels are not targets.
        images = a[:, 2:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    assert images.dtype == np.uint8 and labels.dtype.kind in 'iu'
    n = CFG['official_sizes'][split]
    assert images.shape == (n, *CFG['image_shape']) and labels.shape == (n,)
    assert np.array_equal(np.bincount(labels, minlength=CFG['classes']),
                          np.asarray(CFG['official_class_counts'][split]))
    return images, labels


def to_pil(image):
    from PIL import Image
    if CFG['dataset_kind'] == 'k49':
        # K49 is upright; only its white-ink/black-background polarity is inverted.
        return Image.fromarray(255 - image, mode='L')
    return Image.fromarray(image, mode='RGB')


def class_names():
    if CFG['dataset_kind'] == 'k49':
        with (ROOT / 'raw/k49_classmap.csv').open(encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert [int(v['index']) for v in rows] == list(range(CFG['classes']))
        return [v['char'] for v in rows]
    with tarfile.open(ROOT / 'raw/cifar-100-binary.tar.gz', 'r:gz') as t:
        names = t.extractfile('cifar-100-binary/fine_label_names.txt').read().decode().splitlines()
    assert len(names) == CFG['classes']
    return names
