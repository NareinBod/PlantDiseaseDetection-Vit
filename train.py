from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import Dataset, DataLoader


@dataclass
class VitConfig:
    num_channels: int =  38
    image_size: int =  224
    patch_size: int =  16

    embd_dim: int =  None

    num_patches: int =  None

    num_classes: int = 38

    batch_size: int = 16

    epochs: int = 30

    learning_rate: float = 1e-5 

    adam_weight_decay: int = 0

    adam_betas: tuple = (0.9, 0.999)

    random_seed: int = 42

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

#Different types of images we have.
MODALITIES = ["color","grayscale","segmented"]

transform = Compose([
    ToTensor(),
    #Normalize the distribution: transforming the data to a more consistent range of distribution.
    Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
#location of the images
DATA_DIR = '\plantvillage-dataset\plantvillage dataset\color'

class MultiModalityDataset(Dataset):
    def __init__(self, samples, modality_transforms):
        '''
        samples: list of (img_path, label_id, modality=color)
        modality_transforms: dict {modality_name: transform}
        '''
        
        self.samples = samples
        self.transforms = modality_transforms
    
    def __len__(self):
        return len(self.samples)