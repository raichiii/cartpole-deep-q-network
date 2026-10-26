from PIL import Image
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def seed_everything(seed=42):

    # Python & Environment
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Numerical Libraries
    np.random.seed(seed)

    # PyTorch CPU & GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups

    # CuDNN Determinism (Slightly reduces execution performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enforce deterministic algorithms where possible
    torch.use_deterministic_algorithms(True)

    return np.random.default_rng(seed=seed)


def make_gif_from_ndarray(frames, fname, duration = 20):

    pil_images = [Image.fromarray(frame) for frame in frames]

    pil_images[0].save(
        f"{fname}.gif",
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0,
    )

def plot_simply(x, fname, xscale = None, yscale = None, xlabel = None, ylabel = None):
    fig, ax = plt.subplots()
    if xscale is not None: ax.set_xscale(xscale)
    if yscale is not None: ax.set_yscale(yscale)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    ax.plot(x)
    fig.savefig(f"jpg/{fname}.jpg", dpi=300, bbox_inches="tight")
    plt.close()
