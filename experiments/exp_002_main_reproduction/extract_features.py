"""Paper-aligned indexing adapter around the author's Omniglot dataset.

Full extraction requires recorded approval of the feature-order correction.
The upstream files are unchanged. This is a new dataset artifact, not a claim
to reproduce the unavailable historical feature files byte for byte.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--num-workers', type=int, default=2)
parser.add_argument('--benchmark-only', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--data-order-approved', action='store_true')
args = parser.parse_args()
if not args.benchmark_only and not args.data_order_approved:
    parser.error('Full extraction requires explicit data-order approval')
sys.path.insert(0, str(args.source.resolve()))

import h5py
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from omniglot_dataset import OmniglotFull, RotateAndFlipDataset, RotateAndFlipTransform


class PaperOrder(Dataset):
    """Rows are (augmentation, original class); columns are 20 exemplars."""
    def __init__(self, original):
        self.original = original

    def __len__(self):
        return len(self.original)

    def __getitem__(self, flat_index):
        row, exemplar = divmod(flat_index, 20)
        augmentation, original_class = divmod(row, 1623)
        upstream_index = (original_class * 20 + exemplar) * 8 + augmentation
        image, label = self.original[upstream_index]
        assert label == original_class
        return image, flat_index


assert torch.cuda.is_available(), 'Encoding probe requires the remote NVIDIA GPU'
torch.set_num_threads(4)
args.output.mkdir(parents=True, exist_ok=True)
transform = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])
raw = OmniglotFull(root=str(args.output/'raw'), transform=transform)
assert raw.total_classes == 1623 and len(raw) == 1623 * 20
dataset = PaperOrder(RotateAndFlipDataset(raw, transform=RotateAndFlipTransform()))
encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
encoder.fc = nn.Identity()
encoder = encoder.to('cuda').eval()
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
torch.cuda.reset_peak_memory_stats()
iterator = iter(loader)
with torch.no_grad():
    images, _ = next(iterator)
    encoder(images.to('cuda'))
    torch.cuda.synchronize()
    start = perf_counter()
    timed = 0
    for _ in range(4):
        images, _ = next(iterator)
        encoder(images.to('cuda')).cpu()
        torch.cuda.synchronize()
        timed += images.shape[0]
    seconds = perf_counter() - start
report = dict(purpose='encoder estimate, not training result', torch=torch.__version__,
              device=torch.cuda.get_device_name(), batch_size=args.batch_size,
              num_workers=args.num_workers, torch_cpu_threads=torch.get_num_threads(),
              timed_images=timed, timed_seconds=seconds,
              seconds_per_image=seconds/timed,
              estimated_full_encoding_seconds=seconds/timed*len(dataset),
              peak_cuda_bytes=torch.cuda.max_memory_allocated())
(args.output/'encoder_benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2), flush=True)
if args.benchmark_only:
    raise SystemExit(0)

provenance = dict(torch=torch.__version__, torchvision=torchvision.__version__,
                  weights='ResNet18_Weights.IMAGENET1K_V1',
                  batch_size=args.batch_size, num_workers=args.num_workers,
                  precision='float32', feature_shape=[12984,20,512],
                  index_adapter='(original_class*20+exemplar)*8+augmentation',
                  author_source_sha256={name:hashlib.sha256((args.source/name).read_bytes()).hexdigest()
                                        for name in ['omniglot_dataset.py','omni_features_extract.py']})
def file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
weights_path=Path(torch.hub.get_dir())/'checkpoints'/'resnet18-f37072fd.pth'
provenance['weights_sha256']=file_sha(weights_path)
provenance['raw_archive_sha256']={path.name:file_sha(path)
                                 for path in sorted((args.output/'raw').rglob('*.zip'))}
provenance['class_selection']='author public script: first 1600 original classes in each augmentation; historical random selection unavailable'
provenance['cuda_matmul_allow_tf32']=torch.backends.cuda.matmul.allow_tf32
provenance['cudnn_allow_tf32']=torch.backends.cudnn.allow_tf32
provenance_path=args.output/'encoding_provenance.json'
if provenance_path.exists():
    if json.loads(provenance_path.read_text()) != provenance:
        raise ValueError('Existing encoding provenance differs; inspect before resuming')
else:
    provenance_path.write_text(json.dumps(provenance,indent=2)+'\n')

all_path = args.output/'omniglot_features_all.h5'
if all_path.exists() and not args.resume:
    raise FileExistsError(f'{all_path}; use --resume explicitly')
with h5py.File(all_path, 'a') as f:
    if 'resnet18/224/feat' not in f:
        f.create_dataset('resnet18/224/feat', shape=(12984,20,512), dtype='float32')
        f.create_dataset('original_class', data=np.tile(np.arange(1623),8))
        f.create_dataset('augmentation', data=np.repeat(np.arange(8),1623))
        f.attrs['completed_vectors'] = 0
        f.attrs['index_adapter'] = '(original_class*20+exemplar)*8+augmentation'
        f.attrs['weights'] = 'ResNet18_Weights.IMAGENET1K_V1'
    completed = int(f.attrs['completed_vectors'])
    assert f['resnet18/224/feat'].shape == (12984,20,512)
    assert f.attrs['index_adapter'] == provenance['index_adapter']
    assert f.attrs['weights'] == provenance['weights']
    assert 0 <= completed <= len(dataset)
    subset = torch.utils.data.Subset(dataset, range(completed, len(dataset)))
    data_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    start = perf_counter()
    with torch.no_grad():
        for images, indexes in data_loader:
            vectors = encoder(images.to('cuda')).cpu().numpy()
            if vectors.shape != (len(indexes),512) or not np.isfinite(vectors).all():
                raise ValueError('Invalid encoder output; stop before committing this batch')
            for index, vector in zip(indexes.tolist(), vectors):
                row, exemplar = divmod(index,20)
                f['resnet18/224/feat'][row,exemplar] = vector
            f.attrs['completed_vectors'] = int(indexes[-1])+1
            f.flush()
            if (int(indexes[-1])+1) % (args.batch_size*100) == 0:
                print('encoded', int(indexes[-1])+1, 'of',len(dataset),flush=True)
    f.attrs['encoding_seconds_this_invocation'] = perf_counter()-start

train = (np.arange(1600)[None,:]+1623*np.arange(8)[:,None]).reshape(-1)
test = (np.arange(1600,1623)[None,:]+1623*np.arange(8)[:,None]).reshape(-1)
orders = [('omniglot_features_reordered.h5', np.concatenate([train,test])),
          ('omniglot_features_norotate.h5',np.arange(1623))]
with h5py.File(all_path,'r') as src:
    for name, order in orders:
        path = args.output/name
        if path.exists():
            raise FileExistsError(f'{path}; retain completed artifact and inspect before retry')
        with h5py.File(path,'w') as dst:
            out = dst.create_dataset('resnet18/224/feat',shape=(len(order),20,512),dtype='float32')
            for row,index in enumerate(order):
                out[row] = src['resnet18/224/feat'][int(index)]
            dst.create_dataset('original_class', data=np.array(src['original_class'])[order])
            dst.create_dataset('augmentation', data=np.array(src['augmentation'])[order])
            dst.attrs['index_adapter'] = src.attrs['index_adapter']
            dst.attrs['weights'] = src.attrs['weights']
hashes={}
for path in [all_path]+[args.output/name for name,_ in orders]:
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    hashes[path.name]=digest.hexdigest()
(args.output/'feature_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
