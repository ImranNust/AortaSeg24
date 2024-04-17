# train_script.py
import argparse
import csv, os
import torch
from monai.data import (DataLoader, CacheDataset, load_decathlon_datalist, decollate_batch)
from monai.transforms import AsDiscrete, Decollated
from monai.metrics import DiceMetric
from utils.dataset import DatasetProcessor
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from utils.trainer_script import Trainer


# Define an argument parser to handle command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, required=True, help="Path to the root directory of the dataset")
parser.add_argument("--saved_model_dir", type=str, required=True, help="Path to the root directory where the best model is saved")
parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file for results")
parser.add_argument("--num_classes", type=int, default=24, help="Number of classes for segmentation")
parser.add_argument("--max_iterations", type=int, default=30000, help="Maximum number of training iterations")
parser.add_argument("--eval_num", type=int, default=500, help="Number of validation images for evaluation")
parser.add_argument("--num_train_images", type=int, default=40, help="Number of training images to use")
parser.add_argument("--patch_size", type=int, default=128, help="Size of patches for training")
parser.add_argument("--spatial_dims", type=int, default=3, help="For 3D data it is 3 for 2D data it is 2")
parser.add_argument("--feature_size", type=int, default=96, help="Initial Filters for SegResNet Model")
parser.add_argument("--num_samples", type=int, default=4, help="Number of Samples per batch")
args = parser.parse_args()  # Parse command-line arguments

def main(args):
    print("*"*100)
    print(args)
    print("*"*100)
    if not os.path.exists(args.saved_model_dir):
        # Create the directory if it doesn't exist
        os.makedirs(args.saved_model_dir, exist_ok=True)
        print(f"Directory '{args.saved_model_dir}' created successfully.")
    ########################### DATASET PREPARATION ####################################
    processor = DatasetProcessor(args.root_dir, args.num_train_images)
    # Get train and validation files
    train_files = processor.get_train_files()
    val_files = processor.get_val_files()
    # Get train and validation transforms
    train_transforms = processor.get_train_transforms(args.patch_size, args.num_samples)
    val_transforms = processor.get_val_transforms()
    # Setup Data Loaders
    num_cpus = torch.get_num_threads()
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_num=len(train_files), cache_rate=1.0, num_workers=num_cpus)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=len(val_files), cache_rate=1.0, num_workers=num_cpus)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=num_cpus, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_cpus, pin_memory=True)
    print("Dataset is loaded and prepared for training and validation...")
    print(f"Training Dataset: {len(train_loader)} | Validation Dataset: {len(val_loader)}")
    ###################### DEFINING THE MODEL ################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNETR(
        img_size=(args.patch_size, args.patch_size, args.patch_size),
        in_channels=1,
        out_channels=args.num_classes,
        feature_size=args.feature_size,
        use_checkpoint=True,
    )
    
    if torch.cuda.device_count() > 1:
        print('There are more than 1 GPU... Hurray... Parallel Processing')
        model = torch.nn.DataParallel(model)  # Wrap the model for multi-GPU training
    elif torch.cuda.device_count() == 1:
        print('There are only 1 GPU... Loading model onto it')
    else:
        print("No GPU Detected!!!")
    model = model.to(device)
    
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    print("Defined the model and loaded to the appropriate device...")
    print("Defined loss function and optimizer...")
    ###################### DEFINING THE PARAMETERS ################################################
    post_label = AsDiscrete(to_onehot=args.num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.num_classes)
    trainer = Trainer(model=model,
                      loss_function=loss_function,
                      optimizer=optimizer,
                      max_iterations=args.max_iterations,
                      eval_num=args.eval_num,
                      saved_model_dir=args.saved_model_dir,
                      device=device,
                      patch_size=args.patch_size,
                      num_samples=args.num_samples,  # Include missing argument
                      decollate_batch=decollate_batch,  # Include missing argument
                      post_label=post_label,  # Include missing argument
                      post_pred=post_pred,  # Include missing argument
                      dice_metric=dice_metric)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    while global_step < args.max_iterations:
        global_step, dice_val_best, global_step_best = trainer.train(global_step, train_loader, val_loader, dice_val_best, global_step_best)            
    print(f"Global Step: {global_step} | Best Dice: {dice_val_best} | Global Best: {global_step_best}")
    


if __name__ == "__main__":
    main(args)
