"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
from pathlib import Path

from glob import glob
import SimpleITK
import numpy

import torch
from monai.transforms import ScaleIntensityRange
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
import numpy as np
import gc, os

from .resources.SkipSwinNet_v2 import SkipSwinNet
# from resources.SkipSwinNet_v2 import SkipSwinNet


print('All required modules are loaded!!!')

INPUT_PATH = Path("./test/input")
OUTPUT_PATH = Path("./test/output")
RESOURCE_PATH = Path("./resources")



def run():
    # Read the input
    image, spacing, direction, origin = load_image_file_as_array(
        location=INPUT_PATH / "images/ct-angiography",
    )
    
    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    ###############################################################
    # Set the environment variable to handle memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # Defining the model
    spatial_dims = 3
    block_inplanes = (64, 128, 256, 512) 
    layers=(3,4,6,3)
    in_channels = 1
    num_classes = 24
    encoder_channels = [64, block_inplanes[0], block_inplanes[1], block_inplanes[2]]
    feature_size = 96
    norm_name = 'instance'
    model = SkipSwinNet(spatial_dims=spatial_dims, in_channels=in_channels,num_classes=num_classes, encoder_channels = encoder_channels)
    saved_model_path = RESOURCE_PATH / "Fold0_best_metric_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    state_dict = torch.load(saved_model_path)
    # Remove "module." prefix if necessary
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove "module." prefix
        else:
            new_state_dict[k] = v

    # Load the state dictionary into the model
    model.load_state_dict(new_state_dict)

    # Move the model to the appropriate device
    model.to(device)

    # Optional: Wrap with DataParallel if using multiple GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    print("Defined the model and loaded to the appropriate device...")
    ###############################################################
        
    print(f"ct_angiography shape: {image.shape}")
    transform = ScaleIntensityRange(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True)
    image = transform(image).numpy()
    image = torch.from_numpy(image).permute(2, 1, 0).unsqueeze(0).unsqueeze(0).to(device)
    print(f"image shape: {image.shape}")
    # image = image.to(torch.float32).to(device)
   

    model.eval()
    num_samples = 4
    with torch.no_grad():
        val_outputs = sliding_window_inference(image, (patch_size, patch_size, patch_size), num_samples, model)
    print('Done with prediction! Now saving!!!')
    del model # to save some memory
    del image # to save some memory
    del state_dict # to save some memory
    torch.cuda.empty_cache()
    gc.collect()
    pred_label = torch.argmax(val_outputs, dim = 1)
    if pred_label.is_cuda:
        pred_label = pred_label.detach().cpu()
    del val_outputs
    aortic_branches = pred_label.squeeze().permute(2, 1, 0).numpy().astype(np.uint8)
    print(f"Aortic Branches: Min={np.min(aortic_branches)}, Max={np.max(aortic_branches)}, Type={aortic_branches.dtype}")
    ######################### MY LINES END ################################################
    print('Converted to numpy and integer type 8')
    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/aortic-branches",
        array=aortic_branches,
        spacing=spacing, 
        direction=direction, 
        origin=origin,
    )
    print('Saved!!!')
    return 0


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])
    spacing = result.GetSpacing()
    direction = result.GetDirection()
    origin = result.GetOrigin()
    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result), spacing, direction, origin


def write_array_as_image_file(*, location, array, spacing, origin, direction):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tiff to match the expected output
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    image.SetDirection(direction) # My line
    image.SetOrigin(origin)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())



