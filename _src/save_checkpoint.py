import orbax.checkpoint
from flax.training import orbax_utils
from flax.training import train_state


def save_checkpoint(state: train_state.TrainState, path: str, iteration: int):
    ckpt_dir = path

    # Bundle Arguments:
    ckpt = {'model': state}

    # Save Checkpoint as PyTree:
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=10, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        ckpt_dir, orbax_checkpointer, options,
    )
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpoint_manager.save(
        iteration, ckpt, save_kwargs={'save_args': save_args},
    )
