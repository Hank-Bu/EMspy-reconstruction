# EMspy image reconstruction

Neural-network model accompanying *Electromagnetic Spy: Stealing Information
from Shielded Devices*.

The model reconstructs grayscale screen content from multi-frequency
electromagnetic (EM) leakage measurements of an HDMI display link acquired
using EMspy. It fuses in-phase and quadrature (I/Q) features from five
information-bearing frequencies using a residual U-Net with
squeeze-and-excitation channel attention.

This repository contains the model architecture and a synthetic forward-pass
example. The example uses randomly initialized weights. Trained reconstruction
requires a separately obtained checkpoint.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | Reconstruction network architecture |
| `example.py` | Minimal CPU example using synthetic leakage features |
| `requirements.txt` | Runtime dependency |
| `LICENSE` | MIT license |

## Installation

Use Python 3.10 or later. From the repository directory, install PyTorch using
the dependency range below. Select a PyTorch build appropriate for your system.

```sh
python -m pip install -r requirements.txt
```

## Quick start

```sh
python example.py
```

The example maps a tensor of shape `(1, 10, 32, 48)` to `(1, 1, 32, 48)`.
A CPU is sufficient.

Validated with Python 3.10 and PyTorch 2.5.0. The default model has
25,070,240 trainable parameters.

## Input and output

- Model class `ReconstructionUNet`, with default arguments `(10, 1, 64)`.
- Input shape `(N, 10, H, W)`, containing preprocessed real-valued I/Q
  features stacked in frequency order as
  `[I1, Q1, I2, Q2, I3, Q3, I4, Q4, I5, Q5]`.
- The five center frequencies in the manuscript are 148.5, 297.0, 445.5,
  594.0 and 742.5 MHz. Use the frequency order and preprocessing associated
  with the trained checkpoint. The example supplies synthetic features in
  `[0, 1]` directly to the network.
- `H` and `W` must be positive multiples of 16. The manuscript uses paired
  crops of height 1072 and width 1920. Evaluation supports a minimum of
  16 pixels per dimension. Training with BatchNorm additionally requires
  more than one value per channel at the deepest spatial resolution.
- Output shape `(N, 1, H, W)`, representing grayscale screen content with
  a sigmoid output in `[0, 1]`.

```python
import torch
from model import ReconstructionUNet

model = ReconstructionUNet().eval()
leakage_features = torch.rand(1, 10, 32, 48)
with torch.inference_mode():
    reconstruction = model(leakage_features)
```

## Architecture

A 5-by-5 convolution maps the input to 64 feature channels. Four residual
downsampling stages reduce the spatial resolution to `H / 16` by `W / 16`,
with channel widths of 64, 128, 256 and 512. Two residual blocks form the
bottleneck. The decoder uses residual blocks, transposed convolutions and
convolution-transformed skip connections to restore the spatial resolution.
A final 1-by-1 convolution and sigmoid produce the grayscale image.

## Loading weights

Load a matching state dictionary using strict checking. Supported checkpoint
layouts are a plain state dictionary or a dictionary containing a `model` entry.

```python
checkpoint = torch.load("checkpoint.pth", map_location="cpu", weights_only=True)
state_dict = checkpoint.get("model", checkpoint)
model.load_state_dict(state_dict, strict=True)
model.eval()
```

## License

MIT. See `LICENSE`.
