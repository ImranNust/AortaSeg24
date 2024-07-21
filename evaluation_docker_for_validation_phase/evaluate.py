"""
The following is a simple example evaluation method.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the evaluation, reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
import json
from glob import glob
import SimpleITK
from multiprocessing import Pool
from statistics import mean
from pathlib import Path
from pprint import pformat, pprint

############# MY IMPORTS #################
import torch # My line
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete # My line
import numpy as np # My Line
##########################################

print('Imported all the packages!!!') # deleted

INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
GROUND_TRUTH_DIRECTORY = Path("ground_truth")


def main():
    print_inputs()

    metrics = {}
    predictions = read_predictions()

    # We now process each algorithm job for this submission
    # Note that the jobs are not in any order!
    # We work that out from predictions.json

    # Start a number of process workers, using multiprocessing
    # The optimal number of workers ultimately depends on how many
    # resources each process() would call upon
    with Pool(processes=1) as pool:
        metrics["results"] = pool.map(process, predictions)

    # Now generate an overall score(s) for this submission
    metrics["aggregates"] = {
        # Dice Scores
        "Average_Dice_Score": mean(result["Average_Dice_Score"] for result in metrics["results"]),
        "Zone_0_Dice_Score": mean(result["Zone_0_Dice_Score"] for result in metrics["results"]),
        "Innominate_Dice_Score": mean(result["Innominate_Dice_Score"] for result in metrics["results"]),
        "Zone_1_Dice_Score": mean(result["Zone_1_Dice_Score"] for result in metrics["results"]),
        "Left_Common_Carotid_Dice_Score": mean(result["Left_Common_Carotid_Dice_Score"] for result in metrics["results"]),
        "Zone_2_Dice_Score": mean(result["Zone_2_Dice_Score"] for result in metrics["results"]),
        "Left_Subclavian_Artery_Dice_Score": mean(result["Left_Subclavian_Artery_Dice_Score"] for result in metrics["results"]),
        "Zone_3_Dice_Score": mean(result["Zone_3_Dice_Score"] for result in metrics["results"]),
        "Zone_4_Dice_Score": mean(result["Zone_4_Dice_Score"] for result in metrics["results"]),
        "Zone_5_Dice_Score": mean(result["Zone_5_Dice_Score"] for result in metrics["results"]),
        "Zone_6_Dice_Score": mean(result["Zone_6_Dice_Score"] for result in metrics["results"]),
        "Celiac_Artery_Dice_Score": mean(result["Celiac_Artery_Dice_Score"] for result in metrics["results"]),
        "Zone_7_Dice_Score": mean(result["Zone_7_Dice_Score"] for result in metrics["results"]),
        "SMA_Dice_Score": mean(result["SMA_Dice_Score"] for result in metrics["results"]),
        "Zone_8_Dice_Score": mean(result["Zone_8_Dice_Score"] for result in metrics["results"]),
        "Right_Renal_Artery_Dice_Score": mean(result["Right_Renal_Artery_Dice_Score"] for result in metrics["results"]),
        "Left_Renal_Artery_Dice_Score": mean(result["Left_Renal_Artery_Dice_Score"] for result in metrics["results"]),
        "Zone_9_Dice_Score": mean(result["Zone_9_Dice_Score"] for result in metrics["results"]),
        "Zone_10_R_(Right_Common_Iliac_Artery)_Dice_Score": mean(result["Zone_10_R_(Right_Common_Iliac_Artery)_Dice_Score"] for result in metrics["results"]),
        "Zone_10_L_(Left_Common_Iliac_Artery)_Dice_Score": mean(result["Zone_10_L_(Left_Common_Iliac_Artery)_Dice_Score"] for result in metrics["results"]),
        "Right_Internal_Iliac_Artery_Dice_Score": mean(result["Right_Internal_Iliac_Artery_Dice_Score"] for result in metrics["results"]),
        "Left_Internal_Iliac_Artery_Dice_Score": mean(result["Left_Internal_Iliac_Artery_Dice_Score"] for result in metrics["results"]),
        "Zone_11_R_(Right_External_Iliac_Artery)_Dice_Score": mean(result["Zone_11_R_(Right_External_Iliac_Artery)_Dice_Score"] for result in metrics["results"]),
        "Zone_11_L_(Left_External_Iliac_Artery)_Dice_Score": mean(result["Zone_11_L_(Left_External_Iliac_Artery)_Dice_Score"] for result in metrics["results"])
    }

    # Make sure to save the metrics
    write_metrics(metrics=metrics)

    return 0


def process(job):
    # Processes a single algorithm job, looking at the outputs
    report = "Processing:\n"
    report += pformat(job)
    report += "\n"


    # Firstly, find the location of the results
    aortic_branches_location = get_file_location(
            job_pk=job["pk"],
            values=job["outputs"],
            slug="aortic-branches",
        )
    

    # Secondly, read the results
    aortic_branches = load_image_file(
        location=aortic_branches_location,
    )
    
    print('Aortic Branches are Read Succesfully!')
    print(f"Aortic Branches Shape: {aortic_branches.shape}")


    # Thirdly, retrieve the input image name to match it with an image in your ground truth
    ct_angiography_image_name = get_image_name(
            values=job["inputs"],
            slug="ct-angiography",
    )
    

    # Fourthly, your load your ground truth
    ######################### MY LINES HERE ################################
    ground_truth_segmentation_name = ct_angiography_image_name.split('_')[0]+"_label.mha"
    print(f"Processing: {ground_truth_segmentation_name}")
    # aortic_branches = torch.from_numpy(aortic_branches).permute(2,1,0).unsqueeze(0)
    aortic_branches = np.expand_dims(np.transpose(aortic_branches, axes=(2, 1, 0)), axis=0)
    # Reading Ground Truth Image
    ground_truth_branches = load_ground_truth_file(
        location=GROUND_TRUTH_DIRECTORY,
        ground_truth_segmentation_name = ground_truth_segmentation_name,
    )
    # ground_truth_branches = torch.from_numpy(ground_truth_branches).permute(2,1,0).unsqueeze(0)
    ground_truth_branches = np.expand_dims(np.transpose(ground_truth_branches, axes=(2, 1, 0)), axis=0)
    num_classes = 24
    post_label = AsDiscrete(to_onehot=num_classes, dtype = torch.int8)
    ground_truth_labels = [post_label(ground_truth_branches)]
    del ground_truth_branches
    predicted_labels = [post_label(aortic_branches)]
    del aortic_branches, post_label
    
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    dice_score = dice_metric(y_pred=predicted_labels, y=ground_truth_labels).astype(float)
    del predicted_labels, ground_truth_labels
    print(f"Dice Score is Computed, {dice_score.dtype}")
    average_dice = np.mean(dice_score)
    ####################### MY MODIFICATION END HERE ########################
    # print(report)


    # Finally, calculate by comparing the ground truth to the actual results
    return {
        # Dice Scores
        "Average_Dice_Score": average_dice,
        # "average_nsd": average_nsd,
        "Zone_0_Dice_Score": dice_score[0][0],
        "Innominate_Dice_Score": dice_score[0][1],
        "Zone_1_Dice_Score": dice_score[0][2],
        "Left_Common_Carotid_Dice_Score": dice_score[0][3],
        "Zone_2_Dice_Score": dice_score[0][4],
        "Left_Subclavian_Artery_Dice_Score": dice_score[0][5],
        "Zone_3_Dice_Score": dice_score[0][6],
        "Zone_4_Dice_Score": dice_score[0][7],
        "Zone_5_Dice_Score": dice_score[0][8],
        "Zone_6_Dice_Score": dice_score[0][9],
        "Celiac_Artery_Dice_Score": dice_score[0][10],
        "Zone_7_Dice_Score": dice_score[0][11],
        "SMA_Dice_Score": dice_score[0][12],
        "Zone_8_Dice_Score": dice_score[0][13],
        "Right_Renal_Artery_Dice_Score": dice_score[0][14],
        "Left_Renal_Artery_Dice_Score": dice_score[0][15],
        "Zone_9_Dice_Score": dice_score[0][16],
        "Zone_10_R_(Right_Common_Iliac_Artery)_Dice_Score": dice_score[0][17],
        "Zone_10_L_(Left_Common_Iliac_Artery)_Dice_Score": dice_score[0][18],
        "Right_Internal_Iliac_Artery_Dice_Score": dice_score[0][19],
        "Left_Internal_Iliac_Artery_Dice_Score": dice_score[0][20],
        "Zone_11_R_(Right_External_Iliac_Artery)_Dice_Score": dice_score[0][21],
        "Zone_11_L_(Left_External_Iliac_Artery)_Dice_Score": dice_score[0][22]
    }



def print_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    input_files = [str(x) for x in Path(INPUT_DIRECTORY).rglob("*") if x.is_file()]

    print("Input Files:")
    pprint(input_files)
    print("")


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    with open(INPUT_DIRECTORY / "predictions.json") as f:
        return json.loads(f.read())


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_image_file(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)

def load_ground_truth_file(location, ground_truth_segmentation_name):
    # Use SimpleITK to read a file
    input_files = location / ground_truth_segmentation_name
    result = SimpleITK.ReadImage(input_files)

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    raise SystemExit(main())