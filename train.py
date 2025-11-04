import os
from pathlib import Path
import json
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Resize, RandomHorizontalFlip, RandomRotation, ColorJitter
from sklearn.model_selection import train_test_split
from PIL import Image
from matplotlib import pyplot as plt

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

if os.path.exists(path):
    print("Valid Path")
else:
    print("Invalid Path")
    
#This list will store full paths to every image
all_image_paths = []
#In the same order as teh above list it will store those image's label
all_labels = []

class_folders = os.listdir(path)
class_folders.sort()

# print(f"\nFound {len(class_folders)} class folders")
# print(f"First 3 folders: {class_folders[:3]}")

total_jpg_images = 0
#class folder path to store paths of folder in a sorted way
for class_folder in class_folders:
    if class_folder not in class_to_idx:
        print(f"Warning: {class_folder} not in class_to_idx.json")
        continue
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

# print("Total Image paths: ",len(all_image_paths))
# print("Total Image labels: ",len(all_labels))
# print("Total JPG images: ", total_jpg_images)

#Splitting the Data

'''
Training: 80%
Validation: 10%
Testing: 10%

We'll be using stratified splitting because it uses same proportions of data from all the classes
while random split does'nt do this.

stratify parameter tells the function to lookl at all lables and keep the same proportions
'''
train_paths, temp_paths, train_labels, temp_labels = train_test_split(all_image_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

#Now split the temp_paths and temp_labels into val and test datasets
val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels,stratify=temp_labels, test_size=0.5,random_state=42)

# print("Train paths:", len(train_paths))
# print("Temp paths:", len(temp_paths))
# print("Train labels:", len(train_labels))
# print("Temp labels:", len(temp_labels))
# print("\nAfter second split:")
# print("Val paths:", len(val_paths))
# print("Test paths:", len(test_paths))
# print("Val labels:", len(val_labels))
# print("Test labels:", len(test_labels))
# print(len(train_paths) + len(val_paths) + len(test_paths))

train_transforms = transforms.Compose([
    Resize((224,224)),
    RandomRotation(15),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ToTensor(),
    #These are ImageNet stats-the values ViT was pre-trained on
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#For evaluation we dont want the three(Random rotation, random horizontal flip, and color jitter)
# because we onlyneed variety and randomness for training, not for eval and testing.
#Also we wil use same transform for test as well
eval_transforms = transforms.Compose([
    Resize((224,224)),
    ToTensor(),
    #These are ImageNet stats-the values ViT was pre-trained on
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

'''
Right now we have list of image paths, lables, and transform pipelines ready.

Now we have to load images from those paths, apply transform, pair them with labels, and feed them to dataloaders.
'''
class PlantDiseaseDataset(Dataset):
    def __init__(self, all_image_paths, all_labels, transform= None):
        self.image_paths = all_image_paths
        self.image_labels = all_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, idx):
        img_path =  self.image_paths[idx]
        img_label = self.image_labels[idx]

        image = Image.open(img_path)
        image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        #return them as a tuple.
        return image, img_label
    
#Store the data into the variables
train_data = PlantDiseaseDataset(train_paths, train_labels, train_transforms)
val_data = PlantDiseaseDataset(val_paths, val_labels, eval_transforms)
test_data = PlantDiseaseDataset(test_paths, test_labels, eval_transforms)


#set batch size
batch_size = 32

#create Dataloaders to load the data 32 at a time
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

iterator = iter(train_dataloader)
images, labels = next(iterator)

#Print the shape
print(images.shape, labels.shape)
#we get the shape as ([32, 3, 224, 224]), ([32])

#Visualizing data
unique_labels, counts = np.unique(all_labels, return_counts=True)


plt.figure(figsize=(16, 10))  # Width = 16 inches, Height = 10 inches
plt.barh(class_folders, counts, color="teal")
plt.title("No of Images (Counts)")
plt.xlabel("Plant Disease Class")
plt.ylabel("No of Images")
plt.gca().invert_yaxis()
plt.tick_params(axis='y', labelsize=8)
plt.tight_layout()
plt.savefig("class_vs_images.png")
plt.show()
