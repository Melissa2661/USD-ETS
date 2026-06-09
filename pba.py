from html import entities
from render_usd_2 import UsdRenderer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import h5py as h5
import igl
import os
import numpy as np

fem_loc = "FemElastoDynamics"
fem_mesh_loc = "pbat.fem.Mesh"

def load_pba_simulation(file: str, frame_stop: int, override_fps: bool = False):
    frames = {}
    E = None
    lame_mu = None
    fps = 0
    with h5.File(file, "r") as f:
        fem_data = f[fem_loc]
        sim_data = f["sim"]
        if override_fps:
            fps = f["params"]["fps"]
        E = np.array(fem_data["E"])
        lame_mu = np.array(fem_data["lame_mu"])
        
        for i, frame in enumerate(sim_data):
            if frame_stop != -1 and i > frame_stop:
                break
            X = np.array(sim_data[frame]["x"])
            frames[i] = X
    return E, lame_mu[np.newaxis, :], frames, fps


def render_pba_simulation_to_usd(renderer: UsdRenderer, folder: str, fps: int, frame_stop: int, override_fps: bool, map="Blues"):
    E, lame_mu, frames, file_fps = load_pba_simulation(folder, frame_stop)
    if override_fps:
        fps = file_fps
    dt = 1.0 / fps
    time = 0.0
    cmap = plt.get_cmap(map)

    
    uniques = np.unique(lame_mu)

    log_vals = np.log(uniques)
    # Here, we assume that the Young's modulus set in the scenes 
    # is between 10^5 and 10^8, which is equivalent to the 
    # lame_mu values below
    min_val = np.log(3.4 * 10**4)
    max_val = np.log(3.5 * 10**7)
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    colors = cmap(norm(log_vals))
    # pick elements from the E where val matches the lame_mu value
    submeshes = []
    for i, val in enumerate(uniques):
        selection = np.where(lame_mu == val)[1]
        slice = E[selection, :]
        T, _, _ = igl.boundary_facets(slice)
        submeshes.append((T,))

    for frame_num, frame in enumerate(frames.keys()):
        time += dt
        print(f"Rendering frame {frame} at time {time:.2f}s")
        
        renderer.begin_frame(time)
        X = frames[frame]
        for i, submesh in enumerate(submeshes):
            colour = [colors[i, :3]] if frame_num == 0 else None
            if frame_num == 0:
                print(colour)
            T = submesh[0]
            renderer.render_mesh(f"conncomp_{i}", X, T, colors=colour)
        
        # renderer.render_points("sim", V, radius=0.01)

        renderer.end_frame()
    
