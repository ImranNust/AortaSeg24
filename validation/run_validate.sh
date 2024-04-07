#!/bin/bash

# Load conda module if necessary
module load conda # It is assumed that conda module is already installed!

conda activate AortaSeg24_Challenge

# Necessary packages
packages=("torch" "torchvision" "monai" "SimpleITK" "pandas" "tqdm" "typing-extensions")

# Loop through each package and display its version
for package in "${packages[@]}"
do
    echo "Package: $package"
    pip show $package | grep Version
    echo "-----------------------------------"
done


# Directories
ROOT_DIR="./validation_data/"  # Root directory containing the dataset
SAVED_MODEL_PATH="./best_model/best_metric_model.pth" # Path of the trained model with the best weights
RESULT_DIR="./results/" # Directory where the output segmentations and the quantative results are stored


# Validation Parameters
NUM_CLASSES=24                            # Number of output classes
NUM_SAMPLES=4                             # Number of samples for evaluation


# Run the validation script with the specified parameters
python validate.py \
    --root_dir "$ROOT_DIR" \
    --saved_model_path "$SAVED_MODEL_PATH" \
    --result_dir "$RESULT_DIR" \
    --num_classes "$NUM_CLASSES" \
    --num_samples "$NUM_SAMPLES"



# Deactivate environment
conda deactivate

