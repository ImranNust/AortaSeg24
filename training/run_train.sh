#!/bin/bash
module load conda
# # Check if the AortaSeg24 environment exists
# if conda info --envs | grep AortaSeg24_Challenge; then
#     conda activate AortaSeg24_Challenge
#     echo "Environment already exists."
# else
#     # Create the AortaSeg24 environment with Python 3.10
#     conda create -n AortaSeg24_Challenge python=3.10 -y
#     conda activate AortaSeg24_Challenge

#     # Install the specified packages
#     pip3 install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
#     pip3 install monai[all]
#     pip3 install SimpleITK
#     pip3 install pandas
#     pip3 install tqdm
#     pip3 install typing-extensions
# fi


# Directories
ROOT_DIR="./data/"  # Root directory containing the dataset
SAVED_MODEL_DIR="./best_model/"           # Directory to save the best model

# Output
OUTPUT_CSV="training_info.csv"            # CSV file to save training information

# Training Parameters
NUM_CLASSES=24                            # Number of output classes
MAX_ITERATIONS=30000                      # Maximum number of training iterations
EVAL_NUM=500                              # Frequency of model evaluation
NUM_SAMPLES=4                             # Number of samples for evaluation

# Execution
# ---------

# Run the training script with the specified parameters
# export CUDA_LAUNCH_BLOCKING=1
python train.py \
    --root_dir "$ROOT_DIR" \
    --saved_model_dir "$SAVED_MODEL_DIR" \
    --output_csv "$OUTPUT_CSV" \
    --num_classes "$NUM_CLASSES" \
    --max_iterations "$MAX_ITERATIONS" \
    --eval_num "$EVAL_NUM" \
    --num_samples "$NUM_SAMPLES"
