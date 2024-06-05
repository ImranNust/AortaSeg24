If you are interested in seeing how your Docker image will run and produce segmentation files after submitting to the Grand-Challenge website, this notebook will walk you through those steps.

---

**Please Note: This guide is solely to demonstrate how the `inference.py` code inside the Docker container functions. It is not a tutorial on preparing Docker images!**

---

## Directory Structure

Before we begin, let's first understand the directory structure required for running our Docker container:


```
DockerExample/
├─ resources/                                                   # Folder containing the resouces (your saved model, other utility functions or classes requried to make the prediciton)
|  ├─ your_best_model.pth                                       # Your trained model
|  └── ....                                                     # You can store other dependencies, which you would like to call in the inference.py file to make the prediction
├─ test/
|  ├─── input/images/ct-angiograph/subject001_CTA.mha           # Folder which contains the input for your model to predict the segmentation
│  └─── output/images/aortic-branches/output.mha                # Folder where your predicted segmentations would be stored...
|
├── inference.py                                                # Python script to predict the output segmentation
└── requirements.txt                                            # This file contains the packages required to execute your code              
```

---

## Running the Inference Code Locally

To execute the `inference.py` script locally and generate the segmentation output, follow these steps:

### Step 1: Activate Your Environment

Activate the environment that contains all the necessary dependencies.

```
conda activate Your-environment-with-all-the-packages-installed
```

### Step 2: Run the Inference Script

Execute the `inference.py` script from the terminal:
```
python inference.py
```
---

Upon successful execution, the output segmentation file will be saved directly in the `test/output/images/aortic-branches/` directory. You can then compare this output with the ground-truth output and compute the Dice coefficient and Normalized Surface Distance (NSD) using the MONAI library.

---

## Additional Information

- **Environment Setup:** Ensure that `Your-environment-with-all-the-packages-installed` is properly set up with all the dependencies listed in requirements.txt.
- **Model Preparation:** Place your best-performing model `your_best_model.pth` in the resources/ directory.
- **Input Data:** Store the CT angiography images you wish to segment in the `test/input/images/ct-angiograph/` directory.
- **Output Verification:** After running the inference, verify the segmentation results stored in `test/output/images/aortic-branches/`.

---
**For more detailed instructions on setting up your Docker environment and preparing your model for the Grand-Challenge platform, please refer to the official documentation or tutorials provided by Grand-Challenge.**
