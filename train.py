import os
from pathlib import Path
import json
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

'''
LOADING DATA FROM JSON FILES
'''
try:
    with open("class_to_idx.json",'r') as c:
        class_to_idx = json.load(c)
except  Exception as e:
    print(e)

try:
    with open("idx_to_class.json",'r') as i:
        idx_to_class = json.load(i)
except  Exception as e:
    print(e)

try:
    with open('class_weights.json','r') as cw:
        class_weights = json.load(cw)
except  Exception as e:
    print(e)

'''
BUILD IMAGE PATHS AND LABELS
'''
path = ("plantvillage-dataset\\plantvillage dataset\\color")

if os.path.exists(path=path):
    print("Valid Path")
else:
    print("Invalid Path")
    
#This list will store full paths to every image
all_image_paths = []
#In the same order as teh above list it will store those image's label
all_labels = []

class_folders = os.listdir(path)
class_folders.sort()

print(f"\nFound {len(class_folders)} class folders")
print(f"First 3 folders: {class_folders[:3]}")

total_jpg_images = 0
#class folder path to store paths of folder in a sorted way
for class_folder in class_folders:
    class_idx = class_to_idx[class_folder]

    class_folder_path = os.path.join(path, class_folder)

    if os.path.isdir(class_folder_path):
        image_files = os.listdir(class_folder_path)
    else:
        print("Not a Directory.")
        continue

    print(f"Processing: {class_folder} (idx={class_idx}, {len(image_files)} files)")

    for image_file in image_files:

        if not image_file.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')):
            continue

        total_jpg_images += 1

        image_path = os.path.join(class_folder_path, image_file)

        all_image_paths.append(image_path)
        all_labels.append(class_idx)

print("Total Image paths: ",len(all_image_paths))
print("Total Image labels: ",len(all_labels))
print("Total JPG images: ", total_jpg_images)

#Splitting the Data

'''
Training: 80%
Validation: 10%
Testing: 10%

We'll be using stratified splitting because it uses same proportions of data from all the classes
while random split does'nt do this.

'''