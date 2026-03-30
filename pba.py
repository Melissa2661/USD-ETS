from html import entities
from render_usd import UsdRenderer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import h5py as h5
import igl
import os
import numpy as np

fem_loc = "pbat.sim.dynamics.FemElastoDynamics"
fem_mesh_loc = "pbat.fem.Mesh"

def load_pba_simulation(file: str, frame_stop: int):
    frames = {}
    with h5.File(file, "r") as f:
        data = f["sim"]
        for i, frame in enumerate(data):
            if frame_stop != -1 and i > frame_stop:
                break
            E = np.array(data[frame][fem_loc][fem_mesh_loc]["E"])
            lamegU = np.array(data[frame][fem_loc]["lamegU"])
            X = np.array(data[frame][fem_loc]["x"])
            frames[i] = (X.T, E.T, lamegU.T)
    # print(frames)
    return frames


def render_pba_simulation_to_usd(renderer: UsdRenderer, folder: str, fps: int, frame_stop: int, map="Blues"):
    frames = load_pba_simulation(folder, frame_stop)
    dt = 1.0 / fps
    time = 0.0
    cmap = plt.get_cmap(map)
    for frame in frames.keys():
        time += dt
        renderer.begin_frame(time)

        X, E, lamegU = frames[frame]
        uniques = np.unique(lamegU[:, 0])

        log_vals = np.log(uniques)
        norm = mcolors.Normalize(vmin=np.min(log_vals), vmax=np.max(log_vals))
        colors = cmap(norm(log_vals))

        for i, val in enumerate(uniques):
            slice = E[np.where(lamegU[:, 0] == val)]
            renderer.render_tetmesh(f"sim_{i}", X, slice, update_topology=True, colors=colors)
        #renderer.render_points("sim", V, radius=0.01)
        renderer.end_frame()
    
