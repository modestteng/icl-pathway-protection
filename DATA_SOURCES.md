# Data and model identities used in the manuscript

These pointers document the fixed study inputs; no new download or encoding was
performed for this package. Large images, feature HDF5 files and model checkpoints
are not in this saved-result review archive. Dataset terms continue to apply.

- **Omniglot / original training and mechanism task:** source model and data
  preparation code is included in `omniglot_dataset.py` and
  `omni_features_extract.py`. Original asset hashes/configurations are recorded
  in each experiment's `input_spec.json` and `expected_inputs.json`. The study
  uses the recorded feature identity; an arbitrary similarly named HDF5 is not
  interchangeable. A public source is https://github.com/brendenlake/omniglot .
- **EMNIST Balanced:** the actual frozen source is Hugging Face
  `Royc30ne/emnist-balanced`, revision
  `e6fd3608dc5d46960a89a423472f67c9c0f9ec31`, recorded in exp041/settings.json.
  The study adapts orientation/polarity and uses the registered train/test pools.
- **Kuzushiji-49:** registered CODH archive URLs and source hashes are in each
  `exp043/multiseed_10M_20261022/s18...s22/settings.json`; for example
  https://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz .
- **CIFAR-100:** registered authors' binary archive is
  https://cave.cs.toronto.edu/kriz/cifar-100-binary.tar.gz . Fine classes and the
  study's grayscale encoding are fixed parts of the evaluated condition.
- **Encoder:** frozen torchvision ResNet18 `IMAGENET1K_V1`, cached filename
  `resnet18-f37072fd.pth`; the experiment code checks the registered complete
  weight SHA rather than trusting only that filename. External processing uses
  the original torch/torchvision and normalization identities in the configs.

The included external `episodes.npz` and per-query score rows permit checking
saved label/position balance and reaggregating recorded scores; they do not
permit independent image encoding, model forwards, or CE reconstruction from
all logits. See `REVIEW_GUIDE.md` for exactly what is checked.
