# ==================================================================================================
# KIDNEYLIFE AI - CT KIDNEY IMAGE CLASSIFIER
# Classifies abdominal CT kidney images into 4 groups: Cyst, Normal, Stone and Tumor

#==================================================================================================


# ==================================================================================================
# PART 1 - LIBRARIES AND SETTINGS
# ==================================================================================================

import random                      # Python's own random numbers - used for the seed and for shuffling images in part 3
import numpy as np                 # works with arrays (lists of numbers) and does the maths
import tensorflow as tf            # the deep learning engine that Keras runs on

# ---- project settings (change them here, and they change everywhere below) ----
RANDOM_SEED = 2025                                  # fixed number, so the split and the training give the same result every run
PICTURE_SIZE = (128, 128)                           # every CT image is resized to 128 x 128 pixels so a normal laptop can train
IMAGES_PER_BATCH = 32                               # the model looks at 32 images at a time before updating itself
MAX_EPOCHS = 20                                     # the most full passes over the training images (early stopping can end it sooner)
TRUST_LEVEL = 0.80                                  # predictions below 80% confidence get a "see a radiologist" warning
KIDNEY_CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]   # the 4 classes, written exactly like the folder names in the dataset

random.seed(RANDOM_SEED)           # fixes the seed for Python's random module
np.random.seed(RANDOM_SEED)        # fixes the seed for numpy
tf.random.set_seed(RANDOM_SEED)    # fixes the seed for tensorflow

print("=" * 100)
print("KidneyLife AI - CT Kidney Image Classifier (Cyst, Normal, Stone, Tumor)")
print("=" * 100)


# ==================================================================================================
# PART 2 - FINDING THE DATASET
# ==================================================================================================

import os                          # works with folders and file paths on the computer
import sys                         # lets us stop the program early if something important is missing

print("Looking for the CT kidney dataset...")

downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")   # the Downloads folder of whoever runs the code (works on any PC name)
dataset_name = "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"              # the folder name that comes out of the Kaggle zip

# the Kaggle zip sometimes puts the dataset inside a second folder with the same name, so we check all the usual places
places_to_check = [
    os.path.join(downloads_folder, dataset_name, dataset_name),            # Downloads/dataset/dataset
    os.path.join(downloads_folder, dataset_name),                          # Downloads/dataset
    os.path.join(downloads_folder, "archive", dataset_name, dataset_name), # Downloads/archive/dataset/dataset
    os.path.join(downloads_folder, "archive", dataset_name)                # Downloads/archive/dataset
]

dataset_folder = None                                          # nothing found yet
for place in places_to_check:                                  # go through each possible location
    if os.path.exists(os.path.join(place, "Cyst")):            # the right folder is the one that has a "Cyst" folder inside
        dataset_folder = place                                 # remember it
        break                                                  # stop looking

if dataset_folder is None:                                     # the dataset is not in any of the places
    print("Dataset not found!!! Unzip the Kaggle dataset into:", downloads_folder)
    sys.exit()                                                 # stop the program, nothing can work without the images

for kidney_class in KIDNEY_CLASSES:                                         # check that all 4 class folders are there
    if not os.path.exists(os.path.join(dataset_folder, kidney_class)):     # one class folder is missing
        print("The", kidney_class, "folder is missing from the dataset!!!")
        sys.exit()                                                          # stop the program

print("Dataset found at:", dataset_folder)

split_folder = os.path.join(downloads_folder, "kidneylife_split")   # where our own train/validation/test folders will be created
train_folder = os.path.join(split_folder, "train")                  # 70% of the images - the model learns from these
validation_folder = os.path.join(split_folder, "validation")        # 15% of the images - checked after every epoch
test_folder = os.path.join(split_folder, "test")                    # 15% of the images - only used at the very end

script_folder = os.path.dirname(os.path.abspath(__file__))                 # the folder this python file is saved in
saved_model_file = os.path.join(script_folder, "kidneylife_model.keras")   # the trained model is saved next to this file
print("=" * 100)


# ==================================================================================================
# PART 3 - SPLITTING THE IMAGES INTO TRAIN, VALIDATION AND TEST FOLDERS
# ==================================================================================================

import shutil                                                  # copies image files from one folder to another
from sklearn.utils.class_weight import compute_class_weight    # works out how important each class should be

print("Splitting the images into train (70%), validation (15%) and test (15%)...")

finished_marker = os.path.join(split_folder, "split_done.txt")   # this file is written when the split finishes, so we only split once

if os.path.exists(finished_marker):                              # the split was already done on an earlier run
    print("The images were already split on an earlier run, using that split")
else:
    if os.path.exists(split_folder):                             # a split that stopped half way is still there
        shutil.rmtree(split_folder)                              # delete it so we start clean

    for kidney_class in KIDNEY_CLASSES:                          # split each class on its own so every set has all 4 classes
        class_folder = os.path.join(dataset_folder, kidney_class)    
        # list only the image files and sort them, so the seeded shuffle gives the same order every time
        image_files = sorted([f for f in os.listdir(class_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        random.shuffle(image_files)                              # mix the images so the split is random

        total = len(image_files)                                 # how many images this class has
        train_stop = int(total * 0.70)                           # the first 70% go to training
        validation_stop = train_stop + int(total * 0.15)         # the next 15% go to validation, the rest go to testing

        groups = {
            "train": image_files[:train_stop],                       # from the start up to train_stop
            "validation": image_files[train_stop:validation_stop],   # from train_stop up to validation_stop
            "test": image_files[validation_stop:]                    # from validation_stop to the end
        }

        for group_name, files in groups.items():                             # for train, validation and test
            target_folder = os.path.join(split_folder, group_name, kidney_class)   
            os.makedirs(target_folder, exist_ok=True)                        # create the folder if it does not exist yet
            for file_name in files:                                          # copy each image into its new folder
                shutil.copy(os.path.join(class_folder, file_name), os.path.join(target_folder, file_name))

        print(kidney_class, ":", total, "images split")

    with open(finished_marker, "w") as marker:                   # write the marker file so the next run skips the split
        marker.write("split complete")

# ---- class weights: there are many more Normal images than Stone images ----
images_per_class = []                                                          # will hold the number of training images in each class
for kidney_class in KIDNEY_CLASSES:
    amount = len(os.listdir(os.path.join(train_folder, kidney_class)))          # count the training images in this class
    images_per_class.append(amount)                                             # add the count to the list
    print("Training images for", kidney_class, ":", amount)

# make one label per training image (0 = Cyst, 1 = Normal, 2 = Stone, 3 = Tumor)
all_train_labels = np.concatenate([np.full(amount, number) for number, amount in enumerate(images_per_class)])

# "balanced" gives small classes a bigger weight so the model does not ignore them
weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(KIDNEY_CLASSES)), y=all_train_labels)
class_importance = {number: float(weight) for number, weight in enumerate(weights)}   # Keras wants a dictionary {class number: weight}

print("Class weights:", {KIDNEY_CLASSES[n]: round(w, 3) for n, w in class_importance.items()})
print("=" * 100)
