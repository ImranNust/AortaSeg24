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
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.test_files = image_and_masks_paths(self.root_dir)
        total_images = len(self.test_files)
    
    def get_test_files(self):
        return self.test_files
    
    
    def get_test_transforms(self):
        return Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False),
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
        ])