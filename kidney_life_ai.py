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
