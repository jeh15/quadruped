from typing import Optional
import os

import jax
import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager
from flax.training.train_state import TrainState


def create_default_metadata() -> dict:
    return {'iteration': 0}


def save_checkpoint(
    manager: CheckpointManager,
    directory: str | os.PathLike,
    train_state: TrainState,
    metadata: dict,
) -> None:
    def _is_or_create_directory(directory):
        if not os.path.isdir(directory):
            os.makedirs(directory)
        return directory

    # Make directory if it doesn't exist:
    directory = _is_or_create_directory(directory)

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
    directory: str | os.PathLike,
    train_state: TrainState,
    metadata: dict = create_default_metadata(),
    restore_iteration: Optional[int] = None,
) -> tuple[TrainState, dict]:
    # Check if directory exists:
    assert os.path.isdir(directory), f"Directory {directory} does not exist."

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
            train_state=ocp.args.StandardRestore(abstract_train_state),
            metadata=ocp.args.JsonRestore(metadata),
        ),
    )

    restored_train_state, restored_metadata = (
        restored.train_state,
        restored.metadata,
    )

    return restored_train_state, restored_metadata
