"""
utils.lattice_animator.py

Contains helper functions for animating and showing the lattice state
Uses ffmpeg, which is assumed to be installed in what ffmpeg_path is set to
"""


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


# Where ffmpeg is installed
ffmpeg_path = "lib/ffmpeg-2025-10-16-git-cd4b01707d-essentials_build/bin/ffmpeg.exe"


# Makes an animation from a given list of frames
def animate_lattice(lattice_frames: list[np.ndarray], interval: int, save_path: str, fps: int=20) -> None:
    fig, ax = plt.subplots()
    im = ax.imshow(lattice_frames[0], cmap="binary", vmin=-1, vmax=1)
    # ax.axis("off")

    # Called each frame by FuncAnimation
    def update(frame_idx):
        im.set_data(lattice_frames[frame_idx])
        ax.set_title(f"Sweep {frame_idx * interval}")
        return [im]

    ani = animation.FuncAnimation(
        fig,
        update,
        interval=1000/fps,
        blit=True,
        repeat=False,

        save_count=len(lattice_frames)
    )
    
    # Save video if requested
    if save_path is not None:
        plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
        ani.save(save_path, writer="ffmpeg", fps=fps)
        print(f"[Animation] Saved to {save_path}")

    plt.close(fig)


# Shows the given frame
def show_lattice(lattice_frame: np.ndarray):
    fig, ax = plt.subplots()
    ax.imshow(lattice_frame, cmap="binary", vmin=-1, vmax=1)
    plt.show()