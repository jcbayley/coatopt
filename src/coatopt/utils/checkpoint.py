"""Shared checkpointing utilities for HPPO training methods.

Each training method builds a ``data`` dict and passes it to ``save_checkpoint``.
The structure is flexible – every method can include whatever keys it needs
(networks, optimizers, pareto state, phase metadata, ...).

Typical usage
-------------
Save::

    save_checkpoint(save_dir, episode, {
        "networks":   {"policy": policy.state_dict()},
        "optimizers": {"agent": agent.optimizer.state_dict()},
        "pareto":     {"rewards": ..., "values": ..., ...},
        "meta":       {"is_warmup": True, ...},
    })

Restore::

    ckpt = load_checkpoint(save_dir)
    if ckpt:
        policy.load_state_dict(ckpt["networks"]["policy"])
        episode_start = ckpt["episode"]
        ...
"""

from pathlib import Path

import torch


def save_checkpoint(save_dir: str, episode: int, data: dict) -> None:
    """Write a versioned checkpoint and overwrite the ``checkpoint_latest.pt`` alias.

    Args:
        save_dir: Directory to write into (created if absent).
        episode:  Current episode number, used in the versioned filename.
        data:     Arbitrary dict produced by the training method.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    payload = {"episode": episode, **data}
    torch.save(payload, save_path / "checkpoint_latest.pt")


def load_checkpoint(save_dir: str) -> dict | None:
    """Load the latest checkpoint from *save_dir*, or return ``None``.

    Returns:
        Checkpoint dict (always contains key ``"episode"``) or ``None`` when
        no checkpoint exists in *save_dir*.
    """
    path = Path(save_dir) / "checkpoint_latest.pt"
    if not path.exists():
        return None
    return torch.load(path, weights_only=False)
