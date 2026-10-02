"""Compare correct and shuffled actions for a trained AC-Video JEPA."""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from eb_jepa.architectures import (
    ImpalaEncoder,
    InverseDynamicsModel,
    RNNPredictor,
)
from eb_jepa.datasets.utils import init_data
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
from eb_jepa.state_decoder import MLPXYHead
from eb_jepa.training_utils import load_checkpoint, load_config


def build_models(cfg, data_config, normalizer, device):
    """Build the AC-JEPA and position probe used by the training entry point."""
    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, cfg.model.henc, cfg.model.dstc),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=cfg.model.dobs,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(cfg.model.dobs, data_config.img_size, data_config.img_size),
    )
    predictor = RNNPredictor(
        hidden_size=encoder.mlp_output_dim,
        final_ln=encoder.final_ln,
    )
    test_input = torch.zeros(
        1, cfg.model.dobs, 1, data_config.img_size, data_config.img_size
    )
    _, feature_dim, _, height, width = encoder(test_input).shape
    idm = InverseDynamicsModel(
        state_dim=height * width * feature_dim,
        hidden_dim=256,
        action_dim=2,
    )
    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=cfg.model.regularizer.cov_coeff,
        std_coeff=cfg.model.regularizer.std_coeff,
        sim_coeff_t=cfg.model.regularizer.sim_coeff_t,
        idm_coeff=cfg.model.regularizer.idm_coeff,
        idm=idm,
        first_t_only=cfg.model.regularizer.first_t_only,
        projector=None,
        spatial_as_samples=cfg.model.regularizer.spatial_as_samples,
        idm_after_proj=cfg.model.regularizer.idm_after_proj,
        sim_t_after_proj=cfg.model.regularizer.sim_t_after_proj,
    )
    jepa = JEPA(encoder, nn.Identity(), predictor, regularizer, SquareLossSeq()).to(
        device
    )
    xy_head = MLPXYHead(encoder.mlp_output_dim, normalizer=normalizer).to(device)
    return jepa, xy_head


@torch.inference_mode()
def run(
    checkpoint: str,
    output_dir: str,
    fname: str = None,
    nsteps: int = 8,
    num_samples: int = 4,
):
    """Save action-sensitivity metrics and a trajectory comparison plot."""
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(fname) if fname else checkpoint.parent / "config.yaml"
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, data_config = init_data(
        env_name=cfg.data.env_name,
        cfg_data={**dict(cfg.data), "batch_size": max(num_samples, 4), "num_workers": 0},
    )
    normalizer = val_loader.dataset.normalizer
    jepa, xy_head = build_models(cfg, data_config, normalizer, device)
    checkpoint_info = load_checkpoint(checkpoint, jepa, device=device)
    if not checkpoint_info.get("resumed"):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if "xy_head_state_dict" not in checkpoint_info:
        raise KeyError("Checkpoint does not contain the position probe weights")
    xy_head.load_state_dict(checkpoint_info["xy_head_state_dict"])
    jepa.eval()
    xy_head.eval()

    x, actions, locations, _, _ = next(iter(val_loader))
    x = x[:num_samples].to(device)
    actions = actions[:num_samples].to(device)
    locations = locations[:num_samples].to(device)
    nsteps = min(nsteps, actions.shape[2], x.shape[2] - 1)

    context = x[:, :, :1]
    correct_actions = actions[:, :, :nsteps]
    permutation = torch.roll(torch.arange(x.shape[0], device=device), shifts=1)
    shuffled_actions = correct_actions[permutation]
    correct_states, _ = jepa.unroll(
        context,
        correct_actions,
        nsteps=nsteps,
        unroll_mode="autoregressive",
        compute_loss=False,
    )
    shuffled_states, _ = jepa.unroll(
        context,
        shuffled_actions,
        nsteps=nsteps,
        unroll_mode="autoregressive",
        compute_loss=False,
    )
    target_states = jepa.encoder(x[:, :, : nsteps + 1])

    correct_error = torch.mean((correct_states[:, :, 1:] - target_states[:, :, 1:]) ** 2)
    shuffled_error = torch.mean(
        (shuffled_states[:, :, 1:] - target_states[:, :, 1:]) ** 2
    )
    metrics = {
        "correct_action_latent_mse": float(correct_error),
        "shuffled_action_latent_mse": float(shuffled_error),
        "shuffled_over_correct_ratio": float(shuffled_error / correct_error),
        "nsteps": nsteps,
        "num_samples": x.shape[0],
    }
    (output_dir / "action_sanity_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    correct_xy = xy_head(correct_states).transpose(1, 2)
    shuffled_xy = xy_head(shuffled_states).transpose(1, 2)
    target_xy = locations[:, :, : nsteps + 1].transpose(1, 2)
    correct_xy = normalizer.unnormalize_location(correct_xy).cpu()
    shuffled_xy = normalizer.unnormalize_location(shuffled_xy).cpu()
    target_xy = normalizer.unnormalize_location(target_xy).cpu()

    figure, axes = plt.subplots(1, x.shape[0], figsize=(4 * x.shape[0], 4), squeeze=False)
    for index, axis in enumerate(axes[0]):
        axis.plot(target_xy[index, :, 0], target_xy[index, :, 1], "k-o", label="GT")
        axis.plot(correct_xy[index, :, 0], correct_xy[index, :, 1], "g-o", label="Correct")
        axis.plot(shuffled_xy[index, :, 0], shuffled_xy[index, :, 1], "r--o", label="Shuffled")
        axis.set_title(f"Sample {index}")
        axis.set_xlim(0, data_config.img_size)
        axis.set_ylim(data_config.img_size, 0)
        axis.set_aspect("equal")
        axis.grid(True, alpha=0.3)
        if index == 0:
            axis.legend()
    figure.suptitle("AC-JEPA: correct vs shuffled actions")
    figure.tight_layout()
    figure.savefig(output_dir / "action_trajectory_comparison.png", dpi=160)
    plt.close(figure)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    fire.Fire(run)
