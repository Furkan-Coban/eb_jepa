"""Create local Video JEPA rollout artifacts from a saved checkpoint."""

import json
from pathlib import Path

import fire
import imageio.v2 as imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eb_jepa.architectures import (
    DetHead,
    Projector,
    ResNet5,
    ResUNet,
    StateOnlyPredictor,
)
from eb_jepa.datasets.moving_mnist import MovingMNISTDet
from eb_jepa.image_decoder import ImageDecoder
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.losses import SquareLossSeq, VCLoss
from eb_jepa.training_utils import load_checkpoint, load_config
from examples.video_jepa.eval import visualize_videos


def build_models(cfg, device):
    """Build the same model and visualization heads used by training."""
    encoder = ResNet5(cfg.model.dobs, cfg.model.henc, cfg.model.dstc)
    predictor_model = ResUNet(2 * cfg.model.dstc, cfg.model.hpre, cfg.model.dstc)
    predictor = StateOnlyPredictor(predictor_model, context_length=2)
    projector = Projector(f"{cfg.model.dstc}-{cfg.model.dstc*4}-{cfg.model.dstc*4}")
    regularizer = VCLoss(cfg.loss.std_coeff, cfg.loss.cov_coeff, proj=projector)
    pred_loss = SquareLossSeq(projector)
    jepa = JEPA(encoder, encoder, predictor, regularizer, pred_loss).to(device)

    pixel_decoder = JEPAProbe(
        jepa,
        ImageDecoder(cfg.model.dstc, cfg.model.dobs),
        nn.MSELoss(),
    ).to(device)
    detection_head = JEPAProbe(
        jepa,
        DetHead(cfg.model.dstc, cfg.model.hpre, cfg.model.dobs),
        nn.BCELoss(),
    ).to(device)
    return jepa, pixel_decoder, detection_head


@torch.inference_mode()
def run(
    checkpoint: str,
    output_dir: str,
    fname: str = None,
    num_samples: int = 2,
):
    """Load a checkpoint and save GT/prediction/detection comparison GIFs."""
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(fname) if fname else checkpoint.parent / "config.yaml"
    cfg = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jepa, pixel_decoder, detection_head = build_models(cfg, device)
    checkpoint_info = load_checkpoint(checkpoint, jepa, device=device)
    if not checkpoint_info.get("resumed"):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if "pixel_decoder_state_dict" not in checkpoint_info:
        raise KeyError("Checkpoint does not contain pixel decoder weights")
    if "detection_head_state_dict" not in checkpoint_info:
        raise KeyError("Checkpoint does not contain detection head weights")

    pixel_decoder.head.load_state_dict(checkpoint_info["pixel_decoder_state_dict"])
    detection_head.head.load_state_dict(checkpoint_info["detection_head_state_dict"])
    jepa.eval()
    pixel_decoder.eval()
    detection_head.eval()

    val_set = MovingMNISTDet(split="val", limit=max(num_samples, 8))
    batch = next(iter(DataLoader(val_set, batch_size=num_samples, shuffle=False)))
    batch = {key: value.to(device) for key, value in batch.items()}
    videos = visualize_videos(
        batch,
        jepa,
        pixel_decoder,
        detection_head,
        num_samples=num_samples,
    )

    artifact_paths = []
    for index, video in enumerate(videos):
        frames = video.transpose(0, 2, 3, 1)
        path = output_dir / f"video_jepa_rollout_{index}.gif"
        imageio.mimsave(path, frames, fps=4, loop=0)
        artifact_paths.append(str(path))

    x = batch["video"]
    encoded = jepa.encoder(x)
    predictions, _ = jepa.unroll(
        x,
        actions=None,
        nsteps=x.shape[2] - 2,
        unroll_mode="parallel",
        compute_loss=False,
        return_all_steps=True,
    )
    shapes = {
        "input_video": list(x.shape),
        "encoded_video": list(encoded.shape),
        "one_step_prediction": list(predictions[0].shape),
        "final_rollout_prediction": list(predictions[-1].shape),
    }
    shapes_path = output_dir / "tensor_shapes.json"
    shapes_path.write_text(json.dumps(shapes, indent=2))
    print(json.dumps({"artifacts": artifact_paths, "shapes": shapes}, indent=2))


if __name__ == "__main__":
    fire.Fire(run)
