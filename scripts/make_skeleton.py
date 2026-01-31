import tifffile as tiff
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation
from tqdm import tqdm
import numpy as np
from pathlib import Path

def generate_tubed_skeleton_numpy(label_vol):
    mask = (label_vol == 1)
    skel = skeletonize(mask)
    tubed_skel = binary_dilation(skel, iterations=1)
    return tubed_skel.astype(np.float32)

def run():
    imgs = Path("data/train_images")
    alls = len([fl for fl in imgs.iterdir()])
    for path in tqdm(imgs.iterdir(), total=alls):
        img = tiff.imread(str(path))
        skel = generate_tubed_skeleton_numpy(img)
        path = Path('data/train_skels') / path.name
        np.save(str(path, skel))


if __name__ == "__main__":
    run()