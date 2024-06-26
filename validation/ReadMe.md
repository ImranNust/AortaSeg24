<h1><center><u>Validation</u></center></h1>

This guide provides instructions on implementing a straightforward 3D segmentation model for validation purposes.

<h2><center>Directory Structure</center></h2>
It is assumed that you are in the validation folder, and the validation folder has the following files and folders inside it!

---
```
validation/
├── validation_data/                       # Folder containing the validation data
│   ├── images/
│   │   ├── subject001_CTA.mha             # Input CTA image
│   │   ├── subject002_CTA.mha             # Input CTA image
│   │   │  
│   │   │     
│   │   # ... (similar structure for other data samples)
│   └── masks/
|       ├── subject001_label.mha           # Input Segmentation
|       ├── subject002_label.mha           # Input Segmentation
|       |
|       |
│       │   # ... (similar structure for other data samples)
|
├── utils/                                 # Folder containing utility functions used during validation
│       ├── init.py                        # Empty file to mark utils as a Python package
│       ├── dataset.py                     # Python script containing functions for loading and processing data
│       ├── testing.py                     # Python script containing functions for the validation of the model
|       └── saving_results.py              # Python script to save the result: segmenation output in nifti and mha format; dice and nsd scores
├── run_validation.sh                      # Bash script to execute validation process (likely calls test.py)
├── run_validate.py                        # Python script containing the core validation logic
└── ...                                    # Other project files (requirements.txt, etc.)
```



**Note:**
- The `validation/` folder holds the scripts used for validating the model and the validation_data folder.
- The `validation_data/` folder stores your validation data, with each subfolder containing the corresponding images and segmentation masks.

---

<h2><center>Running Training</center></h2>
To start the validation of the model, run the following command:

```
./run_validate.sh
```

This script will:

1. Create a virtual environment named "AortaSeg24".
2. Activate the environment.
3. Assuming all the required packages are already installed.
   
**If you encounter issues with the environment creation, you can:**

1. Create your own virtual environment.
2. Install the following dependencies listed in the "Dependencies" section.
3. Run the `./run_validate.sh` script again.

<h3><center>Dependencies</center></h3>

The model was trained using the following Python libraries and versions:

- Python: 3.10
- PyTorch: 2.0.0+cu117
- Torchvision: 0.15.1+cu117
- MONAI: 1.3.0
- SimpleITK: 2.3.1
- Pandas: 2.2.1
- TQDM: 4.66.2
- Typing Extensions: 4.11.0

**Note:** These are the specific versions used during training. Compatibility with newer versions might be possible, but it's recommended to stick to the listed versions for consistency.

---

<h1><center><u><b>Citation</b></u></center></h1>

If you find our code or data useful in your research, please cite our papers:

1. @article{imran2024cis,
  title={CIS-UNet: Multi-Class Segmentation of the Aorta in Computed Tomography Angiography via Context-Aware Shifted Window Self-Attention},
  author={Imran, Muhammad and Krebs, Jonathan R and Gopu, Veera Rajasekhar Reddy and Fazzone, Brian and Sivaraman, Vishal Balaji and Kumar, Amarjeet and Viscardi, Chelsea and Heithaus, Robert Evans and Shickel, Benjamin and Zhou, Yuyin and Shao, Wei},
  journal={arXiv preprint arXiv:2401.13049},
  year={2024}
}

2. @article{krebs2024volumetric,
  title={Volumetric Analysis of Acute Uncomplicated Type B Aortic Dissection Using an Automated Deep Learning Aortic Zone Segmentation Model},
  author={Krebs, Jonathan R and Imran, Muhammad and Fazzone, Brian and Viscardi, Chelsea and Berwick, Benjamin and Stinson, Griffin and Heithaus, Evans and Upchurch Jr, Gilbert R and Shao, Wei and Cooper, Michol A},
  journal={Journal of Vascular Surgery},
  year={2024},
  publisher={Elsevier}
}

---
