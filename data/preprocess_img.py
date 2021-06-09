import sys
import os
import numpy as np
from imageio import imwrite
from tqdm import tqdm

from utils import midi2image


def preprocess(midis_path, img_path, rep):
    index = 0
    for midi_path in tqdm(os.listdir(midis_path)[:1000]):
        metrices = midi2image(os.path.join(
            midis_path, midi_path), rep=rep if rep is not None else 1)
        print(len(metrices))
        for img_arr in metrices:
            img_arr = np.rot90(img_arr).astype(np.uint8)
            imwrite(os.path.join(img_path, f'midi_{index}.png'), img_arr)
            index += 1


if __name__ == '__main__':
    midis_path = sys.argv[1]
    img_path = sys.argv[2]
    rep = int(sys.argv[3])
    preprocess(midis_path, img_path, rep)
