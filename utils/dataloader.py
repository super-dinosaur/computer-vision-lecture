import torch
import torchvision.transforms as transforms
import numpy as np
import os

from torch.utils.data import DataLoader, Dataset
from PIL import Image
from typing import List, Dict, Union, Callable, Optional

class VOCDataset(Dataset):
    def __init__(
            self, 
            root, 
            split:str = "train"
    ):
        self.root = root
        self.img_dir = os.path.join(root, split, "images")
        self.img_files = sorted(os.listdir(self.img_dir))
        self.label_dir = os.path.join(root, split, "masks")
        self.label_files = [x.split(".jpg")[0]+".png" for x in self.img_files]
        self.i_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.img_files[idx])).convert("RGB")
        label = Image.open(os.path.join(self.label_dir, self.label_files[idx]))

        img = img.resize((224,224), Image.BILINEAR)
        label = label.resize((224,224), Image.NEAREST)
        img = self.i_transform(img)
        label = torch.from_numpy(np.array(label)).long() #to tensor and long

        return img, label

def get_dataloader(batch_size=8, split="train", shuffle=True):
    dataset = VOCDataset(root="/home/Cv3022244006/code/data", split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader
    