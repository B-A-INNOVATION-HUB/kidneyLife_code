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


# ==================================================================================================
# PART 4 - LOADING THE IMAGES
# ==================================================================================================

print("Loading the images...")

train_images = image_dataset_from_directory(
    train_folder,                  # read from the train folder
    labels="inferred",             # the label of each image comes from the name of the folder it is in
    label_mode="categorical",      # labels become one-hot, e.g. Stone = [0, 0, 1, 0]
    class_names=KIDNEY_CLASSES,    # keeps the class order the same as our list
    color_mode="grayscale",        # CT scans are black and white, so 1 colour channel is enough
    image_size=PICTURE_SIZE,       # resize every image to 128 x 128
    batch_size=IMAGES_PER_BATCH,   # group the images into batches of 32
    shuffle=True,                  # mix the order so the model does not learn the order of the images
    seed=RANDOM_SEED               # the same mix every run
)

validation_images = image_dataset_from_directory(
    validation_folder,             # read from the validation folder
    labels="inferred",
    label_mode="categorical",
    class_names=KIDNEY_CLASSES,
    color_mode="grayscale",
    image_size=PICTURE_SIZE,
    batch_size=IMAGES_PER_BATCH,
    shuffle=False                  # no shuffling, so the validation score is steady
)

test_images = image_dataset_from_directory(
    test_folder,                   # read from the test folder
    labels="inferred",
    label_mode="categorical",
    class_names=KIDNEY_CLASSES,
    color_mode="grayscale",
    image_size=PICTURE_SIZE,
    batch_size=IMAGES_PER_BATCH,
    shuffle=False                  # no shuffling, so the predictions stay in the same order as the true labels
)

print("Images loaded")
print("=" * 100)

# ==================================================================================================
# PART 5 - CLEANING AND AUGMENTING THE IMAGES
# ==================================================================================================

from keras import layers               # building blocks: rescaling and augmentation here, CNN layers in part 6
from keras.models import Sequential    # stacks layers in order (used for the augmentation here and the CNN in part 6)

print("Normalising and augmenting the images...")

scale_pixels = layers.Rescaling(1.0 / 255)   # changes pixel values from 0-255 to 0-1, which helps the network learn

# augmentation = small random changes, so the model sees a slightly different version of each image every epoch
image_changer = Sequential([
    layers.RandomFlip("horizontal"),         # sometimes mirror the image from left to right
    layers.RandomRotation(0.05),             # rotate a little (up to about 18 degrees)
    layers.RandomZoom(0.1),                  # zoom in or out by up to 10%
    layers.RandomTranslation(0.05, 0.05),    # move the image up/down and left/right by up to 5%
    layers.RandomContrast(0.1)               # make the image a little lighter or darker
], name="image_changer")

AUTO = tf.data.AUTOTUNE   # lets tensorflow choose the best speed settings for this computer

train_images = train_images.cache()   # keep the original training images in memory, so they are only read from disk once
train_images = train_images.map(lambda pics, labels: (scale_pixels(image_changer(pics, training=True)), labels),
                                num_parallel_calls=AUTO)   # augment then normalise - this happens again every epoch
train_images = train_images.prefetch(AUTO)   # get the next batch ready while the current batch is training

# validation and test images are only normalised, never augmented, so the scores are fair
validation_images = validation_images.map(lambda pics, labels: (scale_pixels(pics), labels)).cache().prefetch(AUTO)
test_images = test_images.map(lambda pics, labels: (scale_pixels(pics), labels)).cache().prefetch(AUTO)

print("Images ready")
print("=" * 100)

# ==================================================================================================
# PART 6 - BUILDING AND TRAINING THE CNN 
# ==================================================================================================

from keras import metrics                    # gives us Precision and Recall to track while training
from keras.models import load_model          # opens a model that was trained and saved on an earlier run
from keras.optimizers import Adam            # the optimizer that updates the weights while the model learns
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau   # helpers that run after every epoch (explained below)

kidney_model = None   # no model yet

if os.path.exists(saved_model_file):   # a model was trained and saved on an earlier run
    answer = input("A saved model was found. Type L to load it or T to train a new one:   ").strip().lower()
    if answer == "l":                                  # the user wants the saved model
        kidney_model = load_model(saved_model_file)    # open the saved model, no training needed
        print("Saved model loaded")

if kidney_model is None:   # no saved model, or the user chose to train a new one
    print("Building the Convolutional Neural Network...")

    kidney_model = Sequential(name="kidneylife_cnn")   # an empty model that we fill layer by layer

    kidney_model.add(layers.Input(shape=(PICTURE_SIZE[0], PICTURE_SIZE[1], 1)))   # input: a 128 x 128 image with 1 channel

    # 4 feature blocks: each Conv2D finds patterns (edges, shapes, textures) and each MaxPooling2D halves the image size
    # the number of filters doubles in each block (32, 64, 128, 256) so deeper blocks can learn more detailed patterns
    for filter_count in [32, 64, 128, 256]:
        kidney_model.add(layers.Conv2D(filter_count, (3, 3), padding="same", activation="relu"))   # 3 x 3 filters, relu keeps only positive values
        kidney_model.add(layers.MaxPooling2D((2, 2)))                                              # keep the strongest value in every 2 x 2 area

    kidney_model.add(layers.Flatten())                        # turn the 2D feature maps into one long list of numbers
    kidney_model.add(layers.Dense(128, activation="relu"))    # 128 neurons that combine the patterns to make a decision
    kidney_model.add(layers.Dropout(0.5))                     # randomly switch off half the neurons while training to stop overfitting
    kidney_model.add(layers.Dense(len(KIDNEY_CLASSES), activation="softmax"))   # 4 outputs, one probability per class, they add up to 1

    kidney_model.compile(
        optimizer=Adam(learning_rate=0.001),          # Adam adjusts the weights, 0.001 is the size of each step
        loss="categorical_crossentropy",              # the error measure for choosing one class out of many
        metrics=["accuracy",                          # % of images classified correctly
                 metrics.Precision(name="precision"), # of the images the model called a class, how many really were that class
                 metrics.Recall(name="recall")]       # of the images that really were a class, how many the model found
    )

    kidney_model.summary()   # print a table of the layers and the number of parameters

    training_helpers = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),   # stop if validation loss does not improve for 4 epochs, then go back to the best epoch
        ModelCheckpoint(saved_model_file, monitor="val_loss", save_best_only=True, verbose=1),  # save the model every time it gets better
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)               # halve the learning rate if it gets stuck for 2 epochs
    ]

    print("Training the CNN... (this can take a while on a laptop)")
    kidney_model.fit(
        train_images,                         # the images the model learns from
        validation_data=validation_images,    # the images it is checked on after each epoch
        epochs=MAX_EPOCHS,                    # the most epochs to train for
        class_weight=class_importance,        # pay more attention to the classes with fewer images
        callbacks=training_helpers,           # early stopping, saving and learning-rate reduction
        verbose=1                             # show a progress bar
    )

    print("Training finished. Best model saved at:", saved_model_file)

print("=" * 100)

# ==================================================================================================
# PART 7 - TESTING THE CNN ON UNSEEN IMAGES
# ==================================================================================================

import matplotlib.pyplot as plt    # draws the confusion matrix (and shows the CT images in part 8)
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay   # tools for the test results report

print("Testing the model on images it has never seen...")

scores = kidney_model.evaluate(test_images, verbose=0, return_dict=True)   # loss, accuracy, precision and recall on the test images
print(f"Test Loss: {scores['loss']:.4f}")
print(f"Test Accuracy: {scores['accuracy']:.4f}")
print(f"Test Precision: {scores['precision']:.4f}")
print(f"Test Recall: {scores['recall']:.4f}")

class_probabilities = kidney_model.predict(test_images, verbose=0)   # 4 probabilities for every test image
predicted_classes = np.argmax(class_probabilities, axis=1)           # the class with the highest probability is the prediction

# the true class of every test image, in the same order (because test_images was not shuffled)
true_classes = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_images])

# confusion matrix: rows = true class, columns = predicted class, the diagonal shows correct predictions
results_table = confusion_matrix(true_classes, predicted_classes)
ConfusionMatrixDisplay(results_table, display_labels=KIDNEY_CLASSES).plot(cmap="Greens", values_format="d")
plt.title("KidneyLife AI - Confusion Matrix (Test Images)")
plt.tight_layout()
plt.show()   # close the window to continue the program

# precision, recall and f1-score for every class
print("Classification report:")
print(classification_report(true_classes, predicted_classes, target_names=KIDNEY_CLASSES, digits=4))
print("=" * 100)
