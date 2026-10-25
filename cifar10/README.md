# CIFAR-10 Image Classification (PyTorch)

A small CNN trained from scratch on CIFAR-10, plus a Part 2 improvement: **Option A, a simplified ResNet block**. The baseline and the improved model are trained with identical settings so the comparison is fair.

> **Status:** both models were trained and evaluated (10 epochs each, on a Windows laptop CPU). Results are in the Results section below. The 30-epoch default was not run.

## Project structure

```
cifar10/
  data.py            dataset download, normalisation, augmentation, train/val split, DataLoaders
  models.py          baseline CNN and CNN with residual blocks, ResidualBlock
  train.py           training loop, one-cycle LR schedule, checkpoints, optional mixed precision
  evaluate.py        test accuracy, loss/accuracy curves, misclassified images, confusion matrix
  compare.py         baseline vs residual table and curves -> results/
  utils.py           seeding, device, JSON helpers, class names
  tests/             pytest unit tests for the model components and the data pipeline
  cifar10_colab.ipynb  one-click Colab notebook
  requirements.txt
```

## How to run

**Recommended: Google Colab (free GPU).** Push this folder to GitHub, open `cifar10_colab.ipynb` in Colab (File > Open notebook > GitHub), choose a GPU runtime and run all cells.

**Locally** (needs PyTorch, see https://pytorch.org for the right build; about 2 GB):

```
pip install -r requirements.txt
python -m pytest -q                                   # unit tests
python train.py --model baseline --fake --epochs 2    # 1-minute smoke test with random data
python train.py --model baseline --epochs 30 --amp    # --amp only helps on a CUDA GPU
python train.py --model residual --epochs 30 --amp
python evaluate.py --model baseline
python evaluate.py --model residual
python compare.py
```

On CPU only, a full 30-epoch run takes a long time; use fewer epochs (for example `--epochs 10`) or use Colab. A run that is interrupted can be continued with `--resume`.

Outputs per model go to `runs/<model>/`: `best.pt` (best validation accuracy), `last.pt` (for resuming), `history.json`, `config.json`, `curves.png`, `misclassified.png`, `confusion_matrix.png`, `test_metrics.json`. The comparison goes to `results/`.

## Data pipeline

- CIFAR-10 is downloaded with torchvision. The 50,000 training images are split into **45,000 train / 5,000 validation** with a seeded permutation; the 10,000 test images are only used by `evaluate.py`.
- **Normalisation** with the per-channel mean and standard deviation of the training set.
- **Augmentation** (training only): random crop with 4 pixels of padding and random horizontal flip. The validation and test sets are not augmented.
- **Efficient loading:** `torch.utils.data.DataLoader` with worker processes, `pin_memory` on GPU, `persistent_workers`, and `non_blocking` transfers.

## Models

Both models: three stages with 32, 64 and 128 channels, each followed by 2x2 max-pooling and `Dropout2d(0.1)`, then global average pooling, `Dropout(0.3)` and a `Linear(128, 10)` classification head.

| | Stage contents |
|---|---|
| **Baseline** | 2 x (Conv3x3 - BatchNorm - ReLU) |
| **Residual** | 1 `ResidualBlock`: two 3x3 convs with BatchNorm and a skip connection (a 1x1 conv + BatchNorm shortcut when the channel count changes) |

Design decisions:
- **Same depth and widths in both models**, so the only difference is the skip connection (plus the small 1x1 shortcut in the first stage of the residual model). That makes the comparison clean.
- **BatchNorm** after every convolution and no conv bias (BatchNorm has its own bias). It speeds up and stabilises training.
- **Global average pooling** instead of a large fully connected layer keeps the parameter count small and reduces overfitting.
- **Dropout**: light spatial dropout inside the network, stronger dropout before the classifier.
- Trained **from scratch**, no pre-trained weights.

## Training setup

- Loss: cross-entropy. Metrics: loss and accuracy on train and validation each epoch.
- Optimizer: SGD, momentum 0.9 (Nesterov), weight decay 5e-4 (AdamW is available with `--optimizer adamw`).
- **Learning-rate schedule:** one-cycle (`OneCycleLR`), peak LR 0.1, 25% warm-up, stepped every batch.
- Batch size 128, 30 epochs by default. Optional mixed precision (`--amp`).
- **Checkpoints:** `best.pt` whenever validation accuracy improves, `last.pt` every epoch. The test set is used once, with `best.pt`.
- Hyperparameters are defaults chosen from common practice; they were not tuned.

## Reproducibility

- `set_seed` seeds Python, NumPy and PyTorch (CPU and CUDA); the train/val split and the DataLoader shuffle use seeded generators; DataLoader workers are seeded.
- `--deterministic` turns on deterministic cuDNN for bit-exact repeats (slower). Without it, GPU runs can differ slightly between repeats.
- `requirements.txt` lists the dependencies. The versions used for your final results should be added here after the run.

## Part 2: Option A, residual block

`ResidualBlock` in `models.py` is a simplified ResNet basic block. `python compare.py` produces `results/comparison.md` (parameters, best validation accuracy, test accuracy, test loss, training time) and `results/comparison_curves.png`.

## Results

Run settings: 10 epochs, CPU only (Windows laptop, PyTorch 2.14.1 CPU build, Python 3.13), batch size 128, SGD with one-cycle LR (peak 0.1), seed 42, no mixed precision, hyperparameters not tuned. Commands: `python train.py --model <baseline|residual> --epochs 10`, then `evaluate.py` and `compare.py`.

| Metric | Baseline CNN | + Residual blocks |
|---|---|---|
| Parameters | 288,746 | 299,530 |
| Epochs | 10 | 10 |
| Best validation accuracy | 0.8408 | 0.8342 |
| Test accuracy | 0.8313 | 0.8283 |
| Test loss | 0.4890 | 0.5001 |
| Training time (s) | 1212 | 1283 |

Test accuracy difference (residual - baseline): **-0.30 points**.

**Reading the result honestly**
- In this setup the residual blocks did **not** improve the model: the residual network is 0.30 points lower on test accuracy and 0.66 points lower on validation accuracy, with slightly higher test loss.
- This is one run per model with one seed. On a 10,000-image test set, a 0.3 point gap is 30 images, which is within what different seeds could change. The honest conclusion is "no clear difference here", not "residual is worse" and not "residual is better".
- A possible explanation (a hypothesis, not tested): the network is shallow (6 convolutional layers), and skip connections mainly help very deep networks. Testing it would need a deeper network and several seeds.
- Per-class test accuracy is similar for both models. The hardest classes are cat (0.64 / 0.66), bird (0.71 / 0.70) and dog (0.78 / 0.76); the easiest are automobile, ship and truck (above 0.90). Cats, dogs and birds are also the most confused in the confusion matrices (`runs/<model>/confusion_matrix.png`).
- Only 10 epochs were trained; the validation accuracy was still rising at the end (best epoch 10 for both), so longer training would likely score higher for both models.

Plots: `results/baseline_curves.png`, `results/residual_curves.png`, `results/comparison_curves.png`, and the misclassified images in `results/*_misclassified.png`.

## Optional extensions

- Done: mixed-precision training (`--amp`), unit tests (`tests/`).
- Not done: distributed training, ONNX export, FastAPI/Flask deployment.