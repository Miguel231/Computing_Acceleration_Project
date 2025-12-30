import os
import numpy as np


def load_db(npz_path: str):
    """
    DB format:
      names: (N,) array of strings
      embs:  (N,D) float32 array (L2-normalized embeddings)
    """
    if not os.path.exists(npz_path):
        return [], None  # no embeddings yet

    data = np.load(npz_path, allow_pickle=False)
    names = data["names"].tolist()
    embs = data["embs"].astype(np.float32)
    return names, embs