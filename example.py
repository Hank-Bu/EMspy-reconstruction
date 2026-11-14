"""Run the EMspy reconstruction network on synthetic multi-frequency features."""

import torch

from model import ReconstructionUNet


def main():
    torch.manual_seed(0)
    model = ReconstructionUNet().eval()
    leakage_features = torch.rand(1, 10, 32, 48)

    with torch.inference_mode():
        reconstruction = model(leakage_features)

    assert reconstruction.shape == (1, 1, 32, 48)
    assert torch.isfinite(reconstruction).all()
    print(f"Leakage feature shape = {tuple(leakage_features.shape)}")
    print(f"Grayscale reconstruction shape = {tuple(reconstruction.shape)}")
    print(f"Parameters = {sum(parameter.numel() for parameter in model.parameters()):,}")


if __name__ == "__main__":
    main()
