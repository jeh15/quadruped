from typing import List, Optional, Union

import cv2 as cv
import numpy as np
from tqdm import tqdm

import brax
from brax import base
from brax.io import image
import mujoco


def create_video(
    env: brax.System,
    trajectory: Union[List[base.State], base.State],
    filepath: str = 'output.mp4',
    height: int = 240,
    width: int = 320,
    camera: Optional[str] = None,
) -> None:
    # Setup Animation Writer:
    FPS = int(1 / env.dt)

    # Create and set context for mujoco rendering:
    ctx = mujoco.GLContext(width, height)
    ctx.make_current()

    # Generate Frames:
    frames = image.render_array(
        sys=env.sys,
        trajectory=trajectory,
        height=height,
        width=width,
        camera=camera,
    )

    out = cv.VideoWriter(
        filename=filepath,
        fourcc=cv.VideoWriter_fourcc(*'mp4v'),
        fps=FPS,
        frameSize=(width, height),
        isColor=True,
    )

    num_frames = np.shape(frames)[0]
    for i in tqdm(range(num_frames)):
        out.write(frames[i])

    out.release()
