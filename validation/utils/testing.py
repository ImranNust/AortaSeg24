import torch
import os
import numpy as np
import time
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from .saving_results import saving_volumes_mha_and_nifti

def evaluate_model(val_loader, val_ds, model, patch_size, num_samples, post_label, post_pred, dice_metric, nsd_metric, result_dir):
    device = next(model.parameters()).device  # Get the device on which the model is mounted
    model.eval()
    individual_dices = {}
    individual_nsds = {}
    inference_times = []
    reporting_result = {
        "case": {},
        "aggregates": []
    }
    with torch.no_grad():
        # Initialize reporting result dictionary
        reporting_result = {
            "case": {},
            "aggregates": {
                "dice": {
                    "mean": None,
                    "std": None
                },
                "nsd": {
                    "mean": None,
                    "std": None
                }
            }
        }

        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))

            start_time = time.time()
            val_outputs = sliding_window_inference(val_inputs, (patch_size, patch_size, patch_size), num_samples, model)
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)

            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_score = dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            nsd_score = nsd_metric(y_pred=val_output_convert, y=val_labels_convert)

            file_path = val_ds[i]['image_meta_dict']['filename_or_obj']
            vol_name = os.path.basename(file_path).split('.')[0]
            print(f'Processing Volume: {vol_name}')
            individual_dices[vol_name] = dice_score[0].cpu().numpy().tolist()
            individual_nsds[vol_name] = nsd_score[0].cpu().numpy().tolist()
            print(f"Inference Time (s): {inference_time}")
            # print(f"Individual Dices: {individual_dices[vol_name]}")
            print(f"Mean Dice for {vol_name} is {np.mean(individual_dices[vol_name])}")
            print(f"Mean NSD for {vol_name} is {np.mean(individual_nsds[vol_name])}")
            saving_volumes_mha_and_nifti(test_img=val_inputs,
                                         test_label=val_labels,
                                         test_outputs=val_outputs,
                                         vol_name=vol_name,
                                         result_dir=result_dir)
            # For reporting
            # mean_dice = np.mean(individual_dices[vol_name])
            # std_dice = np.std(individual_dices[vol_name])
            # reporting_result["case"][vol_name] = {
            #     "dice": {
            #         "mean": mean_dice,
            #         "std": std_dice
            #     }
            # }

        # Calculate aggregates for reporting
        all_dice_values = [np.mean(individual_dices[vol]) for vol in individual_dices]
        all_nsd_values = [np.mean(individual_nsds[vol]) for vol in individual_nsds]

        reporting_result["aggregates"]["dice"]["mean"] = np.mean(all_dice_values)
        reporting_result["aggregates"]["dice"]["std"] = np.std(all_dice_values)
        reporting_result["aggregates"]["nsd"]["mean"] = np.mean(all_nsd_values)
        reporting_result["aggregates"]["nsd"]["std"] = np.std(all_nsd_values)

    return individual_dices, individual_nsds, inference_times, reporting_result
