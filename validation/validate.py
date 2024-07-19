import argparse
import torch
import monai
import os
import numpy as np
from monai.data import (DataLoader, 
                        CacheDataset, 
                        load_decathlon_datalist, 
                        decollate_batch)
from monai.networks.nets import SwinUNETR
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric, SurfaceDiceMetric
from monai.inferers import sliding_window_inference
import SimpleITK as sitk
import pandas as pd

from utils.dataset import DatasetProcessor
from utils.saving_results import saving_volumes_as_mha, saving_volumes_as_nifti
from utils.testing import evaluate_model


# Define an argument parser to handle command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, required=True, help="Path to the root directory of the dataset")
parser.add_argument("--saved_model_path", type=str, required=True, help="Path of the best model")
parser.add_argument("--result_dir", type=str, required=True, help="Path where results to be stored")
parser.add_argument("--num_classes", type=int, default=24, help="Number of classes for segmentation")
parser.add_argument("--patch_size", type=int, default=128, help="Size of patches for training")
parser.add_argument("--spatial_dims", type=int, default=3, help="For 3D data it is 3 for 2D data it is 2")
parser.add_argument("--feature_size", type=int, default=96, help="Initial Filters for SegResNet Model")
parser.add_argument("--num_samples", type=int, default=4, help="Number of Samples per batch")
args = parser.parse_args()  # Parse command-line arguments

def main(args):
    print("*"*100)
    print(args)
    print("*"*100)
    if not os.path.exists(args.result_dir):
        # Create the directory if it doesn't exist
        os.makedirs(args.result_dir, exist_ok=True)
        print(f"Directory '{args.result_dir}' created successfully.")
    ########################### DATASET PREPARATION ####################################
    processor = DatasetProcessor(args.root_dir)
    # Get test files
    test_files = processor.get_test_files()
    test_transforms = processor.get_test_transforms()
    # Setup Data Loaders
    num_cpus = torch.get_num_threads()
    test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_num=len(test_files), cache_rate=1.0, num_workers=num_cpus)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_cpus, pin_memory=True)
    print("Dataset is loaded and prepared for testing...")
    print(f"Testing Dataset: {len(test_loader)}")
    ###################### DEFINING THE MODEL ########################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNETR(
        img_size=(args.patch_size, args.patch_size, args.patch_size),
        in_channels=1,
        out_channels=args.num_classes,
        feature_size=args.feature_size,
        use_checkpoint=True,
    )
    # Load the saved model
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print('There are more than 1 GPU... Hurray... Parallel Processing')
        state_dict = torch.load(args.saved_model_path)
        model.load_state_dict(state_dict)
        model = torch.nn.DataParallel(model)  # Wrap the model for multi-GPU training
    elif torch.cuda.device_count() == 1:
        print('There is only 1 GPU... Loading model onto it')
        state_dict = torch.load(args.saved_model_path)
        model.load_state_dict(state_dict)
    else:
        print("No GPU Detected!!! Loading model on CPU")
        state_dict = torch.load(args.saved_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    print("Defined the model and loaded to the appropriate device...")

    ###################### DEFINING THE PARAMETERS ###########################
    post_label = AsDiscrete(to_onehot=args.num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.num_classes)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    class_thresholds = [5] * (args.num_classes - 1) # Excluding background
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    nsd_metric = SurfaceDiceMetric(include_background=False, class_thresholds= class_thresholds)
    individual_dices, individual_nsds, inference_times, reporting_result = evaluate_model(val_loader=test_loader,
                                                       val_ds=test_ds, 
                                                       model=model, 
                                                       patch_size=args.patch_size, 
                                                       num_samples=args.num_samples, 
                                                       post_label=post_label, 
                                                       post_pred=post_pred,
                                                       dice_metric=dice_metric,
                                                       nsd_metric = nsd_metric, 
                                                       result_dir=args.result_dir
                                                      )
    df = pd.DataFrame.from_dict(individual_dices)
    mean_dice = df.mean()
    df.loc["Average"] = mean_dice
    df.to_csv(os.path.join(args.result_dir,"test_dices.csv"))
    
    
    df = pd.DataFrame.from_dict(individual_nsds)
    mean_nsd = df.mean()
    df.loc["Average"] = mean_nsd
    df.to_csv(os.path.join(args.result_dir,"test_NSD.csv"))
    
    df = pd.DataFrame.from_dict(inference_times)
    mean_time = df.mean()
    df.loc["Average"] = mean_time
    df.to_csv(os.path.join(args.result_dir,"inference_times.csv"))
    
    print("*"*30)
    print(reporting_result)
    print("*"*30)
if __name__ == "__main__":
    main(args)
