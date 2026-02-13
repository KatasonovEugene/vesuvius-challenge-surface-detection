import sys
import pathlib

from tqdm.auto import tqdm
import tifffile

def main():
    counts = dict()
    for path in tqdm(pathlib.Path('data/train_images').iterdir(), desc="Processing zip files"):
        volume = tifffile.imread(path)
        counts[volume.shape] = counts.get(volume.shape, 0) + 1

    print(counts) # (256, 256, 256): 47, (320, 320, 320): 738, (384, 384, 384): 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

