import os
import random


ORIGINAL_DATASET_DIR = "Training_old"  # e.g., "data/training"
BASE_OUTPUT_DIR = ""  # The new folder for train/val data
VAL_SPLIT = 0.2
RANDOM_SEED = random.randint(1, 100)

random.seed(RANDOM_SEED)

# Create the base output directories
train_dir = 'Training'
val_dir = 'Validation'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get the list of class subfolders
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

for class_name in class_names:
    print(f"Processing class: {class_name}")
    class_dir = os.path.join(ORIGINAL_DATASET_DIR, class_name)
    all_files = os.listdir(class_dir)
    image_files = [f for f in all_files]

    # Random split for training and validation
    random.shuffle(image_files)
    split_index = int(len(image_files) * (1 - VAL_SPLIT))
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    print(f"  Total Images: {len(image_files)}")
    print(f"  Training: {len(train_files)}")
    print(f"  Validation: {len(val_files)}")

    # Create class subdirectories in the new Training and Validation folders
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Copy training images
    for filename in train_files:
        src_path = os.path.join(class_dir, filename)
        dst_path = os.path.join(train_dir, class_name, filename)
        os.rename(src_path, dst_path)

    # Copy validation images
    for filename in val_files:
        src_path = os.path.join(class_dir, filename)
        dst_path = os.path.join(val_dir, class_name, filename)
        os.rename(src_path, dst_path)

print("\nDataset splitting complete!")
print(f"Training set created at: {train_dir}")
print(f"Validation set created at: {val_dir}")