from typing import Optional
import os

import jax
import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions
from flax.training.train_state import TrainState


def default_checkpoint_options() -> CheckpointManagerOptions:
    options = CheckpointManagerOptions(
        max_to_keep=10,
        save_interval_steps=10,
        create=True,
    )
    return options


def default_checkpoint_metadata() -> dict:
    return {'iteration': 0}


def save_checkpoint(
    manager: CheckpointManager,
    train_state: TrainState,
    metadata: dict,
) -> None:
    # Save Checkpoint:
    manager.save(
        int(metadata['iteration']),
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(train_state),
            metadata=ocp.args.JsonSave(metadata)
        ),
    )


def load_checkpoint(
    manager: CheckpointManager,
    train_state: TrainState,
    metadata: dict = default_checkpoint_metadata(),
    restore_iteration: Optional[int] = None,
) -> tuple[TrainState, dict]:
    # Create abstract train state:
    abstract_train_state = jax.tree_util.tree_map(
        ocp.utils.to_shape_dtype_struct, train_state
    )

    # Load Checkpoint:
    if restore_iteration is None:
        restore_iteration = manager.latest_step()

    restored = manager.restore(
        restore_iteration,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_train_state),
            metadata=ocp.args.JsonRestore(metadata),
        ),
    )

    restored_train_state, restored_metadata = (
        restored.state,
        restored.metadata,
    )

    return restored_train_state, restored_metadata
