from typing import List, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

import brax
from brax import base
from brax.io import image
import mujoco

def create_video(
    sys: brax.System,
    trajectory: Union[List[base.State], base.State],
    height: int = 240,
    width: int = 320,
    camera: Optional[str] = None,
) -> None:
    # Setup Animation Writer:
    FPS = int(1 / sys.dt)
    dpi = 300
    writer_obj = FFMpegWriter(fps=FPS)

    # Setup Figure:
    fig, ax = plt.subplots()
    ax.axis('off')

    # Create and set context for mujoco rendering:
    ctx = mujoco.GLContext(width, height)
    ctx.make_current()

    # Generate Frames:
    frames = image.render_array(
        sys=sys,
        trajectory=trajectory,
        height=height,
        width=width,
        camera=camera,
    )

    with writer_obj.saving(fig, 'output.mp4', dpi):
        for frame in frames:
            ax.imshow(frame)
            writer_obj.grab_frame()
