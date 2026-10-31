#!/usr/bin/env python3
"""Convert the official Er0mangaSeg checkpoint to inference-only TorchScript."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import types
from pathlib import Path

REVISION = "2f1d5210459c751ed716b4887be906dffbb4f034"
CHECKPOINT_SHA256 = "9c0e7ae74d043d6ff94e08408bf8a1c032158d2a86d6f7feb6738fb8843b31a6"


def export(config_path: Path, checkpoint_path: Path, output: Path) -> None:
    import torch

    # Er0mangaSeg's UPerHead does not call custom MMCV operators. mmseg imports
    # every optional operator eagerly, however, so mmcv-lite needs a harmless
    # import-time stub to construct this specific network.
    extension = types.ModuleType("mmcv._ext")
    extension.__file__ = "mmcv/_ext.py"
    extension.__getattr__ = lambda _name: lambda *_args, **_kwargs: None
    sys.modules["mmcv._ext"] = extension

    import mmpretrain.models  # noqa: F401
    import mmseg.models  # noqa: F401
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmseg.registry import MODELS

    init_default_scope("mmseg")
    config = Config.fromfile(str(config_path))
    config.model.backbone.init_cfg = None
    # `use_simple` belongs to Er0mangaSeg's older mmseg fork and only affects
    # training loss. Current mmseg can construct the inference graph without it.
    config.model.decode_head.loss_decode = dict(type="CrossEntropyLoss")
    config.model.auxiliary_head.loss_decode = dict(type="CrossEntropyLoss")
    segmentor = MODELS.build(config.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    segmentor.load_state_dict(checkpoint["state_dict"], strict=True)
    segmentor.eval()

    class Inference(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.backbone = model.backbone
            self.decode_head = model.decode_head

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return self.decode_head(self.backbone(image))

    model = Inference(segmentor).eval()
    example = torch.zeros((1, 3, 1024, 1024), dtype=torch.float32)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, check_trace=False)
        traced = torch.jit.freeze(traced)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(traced, str(output))

    license_source = config_path.parents[2] / "LICENSE"
    if license_source.is_file():
        shutil.copy2(license_source, output.with_name("LICENSE-Er0mangaSeg"))


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    source = repository / "target" / "eromanga" / "source"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=source / "configs" / "convnext" / "convnext_h.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repository
        / "target"
        / "eromanga"
        / "home"
        / "user"
        / "Er0mangaSeg"
        / "pretrained"
        / "convnext_1024_iter_400.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository
        / "target"
        / "release"
        / "store"
        / "eromanga-seg"
        / REVISION
        / "model.pt",
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Er0mangaSeg checkpoint not found: {args.checkpoint}. "
            "Extract convnext_1024_iter_400.pth from er0manga/er0mangademo."
        )
    with args.checkpoint.open("rb") as checkpoint:
        digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError(
            f"Er0mangaSeg checkpoint SHA-256 mismatch: {digest}. "
            "Only the pinned upstream checkpoint is accepted."
        )
    export(args.config, args.checkpoint, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
