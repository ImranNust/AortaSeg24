import os
import glob
from pathlib import Path
from random import seed
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    RandRotate90d,
    Spacingd,
    RandAffined
)

def image_and_masks_paths(root_dir):
    # Initialize empty lists to store image and segmentation file paths
    image_paths = []
    segmentation_paths = []

    # Iterate through all subdirectories within the root directory
    for subdir, dirs, files in os.walk(root_dir):
        # Check for the "images" subdirectory
        images_subdir = os.path.join(subdir, "images")
        if os.path.isdir(images_subdir):
            # Find the image file within the "images" subdirectory
            for filename in os.listdir(images_subdir):
                if filename.lower().endswith((".mha")):  # Adjust extensions as needed
                    image_path = os.path.join(images_subdir, filename)
                    image_paths.append(image_path)

        # Check for the "masks" subdirectory
        masks_subdir = os.path.join(subdir, "masks")
        if os.path.isdir(masks_subdir):
            # Find the segmentation file within the "masks" subdirectory
            for filename in os.listdir(masks_subdir):
                segmentation_path = os.path.join(masks_subdir, filename)
                segmentation_paths.append(segmentation_path)
    files = [{"image": image_name, "label":label_name} for image_name, label_name in zip(sorted(image_paths), sorted(segmentation_paths))]
    return files

class DatasetProcessor:
    def __init__(self, root_dir, num_train_images=40):
        self.root_dir = Path(root_dir)
        self.files = image_and_masks_paths(self.root_dir)
        total_images = len(self.files)
        if num_train_images >= total_images:
            raise ValueError("Number of training images must be less than the total number of images.")
        self.train_files = self.files[0:num_train_images]
        self.val_files = self.files[num_train_images:]
    
    def get_train_files(self):
        return self.train_files
    
    def get_val_files(self):
        return self.val_files

    def get_train_transforms(self, patch_size = 96, num_samples = 4):
        return Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False),
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            RandCropByPosNegLabeld(keys=["image", "label"], 
                                   label_key="label", 
                                   spatial_size=(patch_size, patch_size, patch_size),
                                   pos=1, neg=1, num_samples=num_samples, 
                                   image_key="image", image_threshold=0),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
            RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
        ])
    
    def get_val_transforms(self):
        return Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False),
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ])
