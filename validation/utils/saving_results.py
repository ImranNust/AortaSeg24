import os
import torch
import SimpleITK as sitk

def saving_volumes_as_mha(test_img, test_label, test_outputs, vol_name, result_dir):
    
    # Set file paths for saving
    img_file = os.path.join(result_dir, vol_name+ ".mha")
    orig_label_file = os.path.join(result_dir, vol_name+ "_original.mha")
    pred_label_file = os.path.join(result_dir, vol_name+ "_predicted.mha")

    # Save test_img
    img1 = test_img.detach().cpu()
    img1 = torch.squeeze(img1)
    img1 = img1.permute(2, 1, 0)
    new_sitk_img = sitk.GetImageFromArray(img1)
    new_sitk_img.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(new_sitk_img, img_file)

    # Save test_label (original)
    lab1 = test_label.detach().cpu()
    lab1 = torch.squeeze(lab1)
    lab1 = lab1.permute(2, 1, 0)
    new_sitk_lab = sitk.GetImageFromArray(lab1)
    new_sitk_lab.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    sitk.WriteImage(new_sitk_lab, orig_label_file)

    # Save predicted label
    pred_label = torch.argmax(test_outputs, dim=1).detach().cpu()[0]
    pred_label = pred_label.permute(2, 1, 0)
    pred_sitk_lab = sitk.GetImageFromArray(pred_label)
    pred_sitk_lab.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    sitk.WriteImage(pred_sitk_lab, pred_label_file)
    
    print(f'Results for {vol_name} saved!!!')


def saving_volumes_as_nifti(test_img, test_label, test_outputs, vol_name, result_dir):
    
    # file_path = test_ds[0]['image_meta_dict']['filename_or_obj']
    # vol_name = os.path.basename(file_path).split('.')[0]
    os.path.join(result_dir, vol_name+ ".nii.gz")

    img1 = test_img.detach().cpu()
    img1 = torch.squeeze(img1)
    img1 = img1.permute(2, 1, 0)
    new_sitk_img = sitk.GetImageFromArray(img1)
    new_sitk_img.SetDirection((-1.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,1.0))
    sitk.WriteImage(new_sitk_img, os.path.join(result_dir, vol_name+ ".nii.gz"))

    lab1 = test_label.detach().cpu()
    lab1 = torch.squeeze(lab1)
    lab1 = lab1.permute(2, 1, 0)
    new_sitk_lab = sitk.GetImageFromArray(lab1)
    new_sitk_lab.SetDirection((-1,0,0,0,-1,0,0,0,1))
    sitk.WriteImage(new_sitk_lab, os.path.join(result_dir, vol_name+ "_original.nii.gz"))

    pred_label = torch.argmax(test_outputs, dim=1).detach().cpu()[0]
    pred_label = pred_label.permute(2, 1, 0)
    pred_sitk_lab = sitk.GetImageFromArray(pred_label)
    pred_sitk_lab.SetDirection((-1,0,0,0,-1,0,0,0,1))
    sitk.WriteImage(pred_sitk_lab, os.path.join(result_dir, vol_name+ "_predicted.nii.gz"))
    
    print(f'Results for {vol_name} saved!!!')

    
def saving_volumes_mha_and_nifti(test_img, test_label, test_outputs, vol_name, result_dir):
    
    # Set file paths for saving
    img_file = os.path.join(result_dir, vol_name+ ".mha")
    orig_label_file = os.path.join(result_dir, vol_name+ "_original.mha")
    pred_label_file = os.path.join(result_dir, vol_name+ "_predicted.mha")

    # Save test_img
    img1 = test_img.detach().cpu()
    img1 = torch.squeeze(img1)
    img1 = img1.permute(2, 1, 0)
    new_sitk_img = sitk.GetImageFromArray(img1)
    new_sitk_img.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    # Set metadata for the label
    new_sitk_img.SetOrigin((0, 0, 0))  # Set origin as (0,0,0)
    new_sitk_img.SetSpacing((1, 1, 1))  # Set spacing as (1,1,1)
    sitk.WriteImage(new_sitk_img, os.path.join(result_dir, vol_name+ ".nii.gz"), True)
    new_sitk_img = sitk.ReadImage(os.path.join(result_dir, vol_name+ ".nii.gz"))
    sitk.WriteImage(new_sitk_img, img_file, True)

    # Save test_label (original)
    lab1 = test_label.detach().cpu()
    lab1 = torch.squeeze(lab1)
    lab1 = lab1.permute(2, 1, 0)
    new_sitk_lab = sitk.GetImageFromArray(lab1)
    new_sitk_lab.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    # Set metadata for the label
    new_sitk_lab.SetOrigin((0, 0, 0))  # Set origin as (0,0,0)
    new_sitk_lab.SetSpacing((1, 1, 1))  # Set spacing as (1,1,1)
    sitk.WriteImage(new_sitk_lab, os.path.join(result_dir, vol_name+ "_original.seg.nrrd"), True)
    new_sitk_lab = sitk.ReadImage(os.path.join(result_dir, vol_name+ "_original.seg.nrrd"))
    sitk.WriteImage(new_sitk_lab, orig_label_file, True)

    # Save predicted label
    pred_label = torch.argmax(test_outputs, dim=1).detach().cpu()[0]
    pred_label = pred_label.permute(2, 1, 0)
    pred_sitk_lab = sitk.GetImageFromArray(pred_label)
    pred_sitk_lab.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    # Set metadata for the predicted label
    pred_sitk_lab.SetOrigin((0, 0, 0))  # Set origin as (0,0,0)
    pred_sitk_lab.SetSpacing((1, 1, 1))  # Set spacing as (1,1,1)
    sitk.WriteImage(pred_sitk_lab, os.path.join(result_dir, vol_name+ "_predicted.seg.nrrd"), True)
    pred_sitk_lab = sitk.ReadImage(os.path.join(result_dir, vol_name+ "_predicted.seg.nrrd"))
    sitk.WriteImage(pred_sitk_lab, pred_label_file, True)
    
    print(f'Results for {vol_name} saved!!!')
