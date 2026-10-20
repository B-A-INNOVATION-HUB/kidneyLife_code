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
