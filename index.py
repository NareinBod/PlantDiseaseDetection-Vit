import os
from pathlib import Path
import json
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Resize, RandomHorizontalFlip, RandomRotation, ColorJitter
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from PIL import Image
from matplotlib import pyplot as plt
import transformers
from transformers import ViTImageProcessor, ViTModel, ViTForImageClassification
import matplotlib.pyplot as plt
import numpy as np

#Plant Dataset
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


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
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

        # print(f"Processing: {class_folder} (idx={class_idx}, {len(image_files)} files)")

        for image_file in image_files:

            if not image_file.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')):
                continue

            total_jpg_images += 1

            image_path = os.path.join(class_folder_path, image_file)

            all_image_paths.append(image_path)
            all_labels.append(class_idx)

    # print("Total Image paths: ",len(all_image_paths))

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

    train_transforms = transforms.Compose([
        Resize((224,224)),
        RandomRotation(15),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor(),
        #These are ImageNet stats-the values ViT was pre-trained on
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    #For evaluation we dont want the three(Random rotation, random horizontal flip, and color jitter)
    # because we onlyneed variety and randomness for training, not for eval and testing.
    #Also we wil use same transform for test as well
    eval_transforms = transforms.Compose([
        Resize((224,224)),
        ToTensor(),
        #These are ImageNet stats-the values ViT was pre-trained on
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    '''
    Right now we have list of image paths, lables, and transform pipelines ready.

    Now we have to load images from those paths, apply transform, pair them with labels, and feed them to dataloaders.
    '''

    #Visualizing data
    unique_labels, counts = np.unique(all_labels, return_counts=True)


    #Store the data into the variables
    train_data = PlantDiseaseDataset(train_paths, train_labels, train_transforms)
    val_data = PlantDiseaseDataset(val_paths, val_labels, eval_transforms)
    test_data = PlantDiseaseDataset(test_paths, test_labels, eval_transforms)


    #set batch size
    batch_size = 64

    #create Dataloaders to load the data 32 at a time
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)


    #Model setup
    num_classes = len(class_folders)

    iterator = iter(train_dataloader)
    images, labels = next(iterator)

    #Transformer already handles pre-processing
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels = num_classes)

    device = ( "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # print(model)

    #Testing to see if the model gives the correct output shape
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)

    #Test if our input shape and output shape is correct?
    # print(f"Input batch shape: {images.shape}")
    # print(f"Output logits shape: {outputs.logits.shape}")

    #we have to pass class weights to loss function as a tensor
    class_weights_tensor = torch.tensor(class_weights).to(device)

    #loss function(criterion) and loss is the value during training
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    #optimizer: updates models paramters after each batch
    optim = torch.optim.AdamW(model.parameters(),lr=3e-5, weight_decay=0.01)

    num_epochs = 10
    best_val_accuracy = 0.0
    #where to store the best version of the model (Model with highest accuracy in a specific epoch)
    best_model_path = "best_vit_plant_disease.pth"

    #The learning rate is updated recursively
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=num_epochs, eta_min=1e-6 )

    #scaler
    scaler = torch.cuda.amp.GradScaler()

    #Train
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch, (images, labels) in enumerate(dataloader):
            #move to device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # # WRAP FORWARD PASS IN AUTOCAST
            with torch.cuda.amp.autocast():
                prediction = model(images)
                loss = loss_fn(prediction.logits, labels)
            
            # USE SCALER FOR BACKWARD
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            predicted_classes = torch.argmax(prediction.logits, dim=1)
            train_correct += (predicted_classes == labels).sum().item()
            train_total += images.size(0)
            train_loss += loss.item()

                # #check progress
                # if batch % 100 == 0:
                #     loss, current = loss.item(), (batch + 1)*len(images)
                #     print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

            if batch % 200 == 0:
                loss, current = loss.item(), (batch + 1) * len(images)
                print(f"Train Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        
        avg_loss  = train_loss/ len(dataloader)
        accuracy = (train_correct / train_total) * 100

        return avg_loss, accuracy

    # print(f"Current Training Device: {device.upper()}")     

    #val: practice data
    def validate(dataloader, model, loss_fn):
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        model.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for batch, (images, labels) in enumerate(dataloader):
                    #move to device
                    images, labels = images.to(device), labels.to(device)

                    prediction = model(images)

                    loss = loss_fn(prediction.logits, labels)
                    predicted_classes = torch.argmax(prediction.logits, dim=1)
                    val_correct += (predicted_classes == labels).sum().item()
                    val_total += images.size(0)
                    val_loss += loss.item()
            
        avg_loss = val_loss / len(dataloader)
        accuracy = (val_correct/ val_total)*100
        return avg_loss, accuracy

    def test(dataloader, model, loss_fn):
        test_loss = 0
        test_correct = 0
        test_total = 0
        model.eval()

        with torch.no_grad():
            for batch, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)

                prediction = model(images)

                loss = loss_fn(prediction.logits, labels)

                predicted_classes = torch.argmax(prediction.logits, dim=1)
                test_correct += (predicted_classes == labels).sum().item()
                test_total += images.size(0)
                test_loss += loss.item()

            avg_loss = test_loss/ (len(dataloader))
            accuracy = (test_correct/test_total) * 100

        return avg_loss, accuracy

    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")
        
        # Training
        train_avg_loss, train_accuracy = train(train_dataloader, model, criterion, optim)
        
        # Validation     
        val_avg_loss, val_accuracy = validate(val_dataloader, model, criterion)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Print summary
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"   Train Loss: {train_avg_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"   Val Loss: {val_avg_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        
        #save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved! (Val Acc: {val_accuracy:.2f}%)")
        
        print("-" * 50)

    print(f"\n Training Complete!")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"Model saved to: {best_model_path}") 

    #Testing

    #load model with highest accuracy
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    test_avg_loss, test_accuracy = test(test_dataloader, model, criterion)

    print(f"\n--- Test Summary ---")
    print(f"Test Loss: {test_avg_loss:.4f} | Test Acc: {test_accuracy:.2f}%")
    print("\nTesting Complete!")

    #Analysis
    all_preds = []
    true_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in (test_dataloader):
            images = images.to(device)
            prediction = model(images)
            predicted_classes = torch.argmax(prediction.logits, dim=1)
            all_preds.extend(predicted_classes.cpu().numpy())
            true_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    true_labels = np.array(true_labels)

    print(f"First 10 predictions: {all_preds[:10]}")
    print(f"First 10 true labels: {true_labels[:10]}")

    print(f"Last 10 predictions: {all_preds[-10:]}")
    print(f"Last 10 true labels: {true_labels[-10:]}")

    # Get all unique class indices that were actually present in the test data
    unique_indices = sorted(np.unique(np.concatenate([true_labels, all_preds])))
    
    # Map these indices to their actual class names using idx_to_class
    # idx_to_class keys are strings, so must convert the indices to strings.
    class_names = [idx_to_class[str(i)] for i in unique_indices]
    
    # confusion matrix:  used to describe the performance of a classification model
    confusion_matrix = metrics.confusion_matrix(true_labels, all_preds)
    
    # Use the full list of class names for display_labels
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix = confusion_matrix, 
        display_labels = class_names
    )
    
    # Increase figure size for better readability of many labels
    fig, ax = plt.subplots(figsize=(20, 20))
    cm_display.plot(ax=ax, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.title(f'Confusion Matrix - Plant Disease Classification', 
              fontsize=18, pad=20, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved to 'confusion_matrix.png'")
    plt.show()
    plt.close()

    #classification report
    print("Classification Report")

    report = classification_report (
        true_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print("\n" + '='*80)
    print("CLASSIFICATION REPORT (Precision, Recall, F1-Score, Support)")
    print("="*80)
    print(report)
    print("="*80)

    # Save the report to a file
    report_file_path = 'classification_report.txt'
    with open(report_file_path, 'w') as f:
        f.write(report)
    print(f"Classification Report saved to '{report_file_path}'")
    
    print("\n" + "="*60)
    print("ALL ANALYSIS COMPLETE!")
    print("="*60)

    