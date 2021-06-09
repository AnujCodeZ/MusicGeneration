import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PianoDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_paths = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        
        img = Image.open(os.path.join(self.data_dir, self.img_paths[idx]))
        t = transforms.ToTensor()
        img = t(img)
        
        return img
