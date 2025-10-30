import os
import torch
import json
# Get the list of all files and directories
path = "plantvillage-dataset\\plantvillage dataset\\color"
dir_list = os.listdir(path)

print("Color Folder Summary: File type is JPG")
dir_list.sort()
total_in_folder = 0
class_counts = []
classes_list = []
for sub_folder in dir_list:
    print(f"{sub_folder}")
    classes_list.append(sub_folder)
    files = os.listdir(f"plantvillage-dataset\\plantvillage dataset\\color\\{sub_folder}")
    print(f"Has {len(files)} files")
    class_counts.append(len(files))
    total_in_folder += len(files)
print("\n")
print("Total in entire folder: ", total_in_folder)
print("\n")
print("Class Counts List: ", class_counts)

#Mean per class:1429 images
#Median per class: 1076 images

#largest class: Orange___Huanglongbing (5507 images)
#smallest class: Potato___healthy (152 images)

#Imbalance ratio: largest/smalles = 36.23

#This imabalance is too high to resolve this 36.23 imabalance ratio we
#Use a weighted loss function
'''
Weighted Loss Function:

It's a modification to a standard loss function. the weights are used to 
assign a higher penality to mis-classifications of minority class.

Goal is to make model more sensitive to minority class by increasing cost of mis classification of that class.
so from the formula if we observe we have number of samples in the denominator so
if we have a class with less images that class's weight will be higher and vice verse.

Finally we pass these weights, converted into a tensor, to our loss function
'''

#For a weighted loss function we have to calculate class weights 

num_classes = 38
weights_list = []
for no_of_samples in class_counts:
    weight = total_in_folder/ (num_classes * no_of_samples)
    weights_list.append(weight)

weights_tensor = torch.tensor(weights_list)
print("\n")
print("Weights for each class: ", weights_tensor)

print(classes_list)

classes_list.sort()

class_to_index = {}
num = 0
for i in classes_list:
    class_to_index[i] = num
    num += 1
print("\n")
print("Class to Index Mapping.")
print((class_to_index))

json_form = json.dumps(class_to_index, indent=3)
print(json_form)

with open("class_to_idx.json", "w") as r:
    r.write(json_form)

with open("class_weights.json", 'w') as w:
    w.write(json.dumps(weights_list, indent=3))