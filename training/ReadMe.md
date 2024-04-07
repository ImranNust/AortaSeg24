<h1><center><u>Training Tutorial</u></center></h1>

This guide provides instructions on implementing a straightforward 3D segmentation model for training purposes.
<h2><center>Directory Structure</center></h2>
It is assumed that you are in the training folder, and the training folder has the following files and folders inside it!

---
```
project_root/
├── data/                                  # Folder containing the training data
│   ├── UF002/
│   │   ├── images/
│   │   │   └── UF002_CTA.mha              # Input CTA image
│   │   └── masks/
│   │       └── UF002_CTA_label.mha        # Output segmentation mask
│   ├── UF003/
│   │   # ... (similar structure for other data samples)
│   └── ...
├── training/                              # Folder containing training scripts
│   ├── run_train.sh                       # Bash script to execute training process (likely calls train.py)
│   ├── train.py                           # Python script containing the core training logic
│   └── utils/                             # Folder containing utility functions used during training
│       ├── init.py                        # Empty file to mark utils as a Python package
│       ├── dataset.py                     # Python script containing functions for loading and processing data
│       └── trainer_script.py              # Python script containing functions for training the model
└── ...                                    # Other project files (requirements.txt, etc.)
```
**Note"**
- The `data/` folder stores your training data, with each subfolder containing the corresponding images and segmentation masks.
- The `training/` folder holds the scripts used for training the model.

---
<h2><center>Running Training</center></h2>
To train the model, run the following command:

```
./run_train.sh
```

This script will:

1. Create a virtual environment named "AortaSeg24".
2. Activate the environment.
3. Install all required dependencies.
   
**If you encounter issues with the environment creation, you can:**

1. Create your own virtual environment.
2. Install the following dependencies listed in the "Dependencies" section.
3. Run the `./run_train.sh` script again.

**Dependencies**

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
<h3><center>Parameters Details</center></h3>

We have split the data into training and validation sets. For example, if you have 60 images and select 50 images for training and 10 images for validation, you need to change the **num_train_images** parameter to 50 in the `utils.datasets.DataProcessor` class. This will automatically reserve 50 images for training and 10 for validation. If, for some reason, you don't want to reserve any images for validation, you'll need to comment out the lines involving validation.

---

<h1><center><u><b>Citation</b></u></center></h1>

If you find our code or data useful in your research, please cite our papers:
1. @article{imran2024cis,
  title={CIS-UNet: Multi-Class Segmentation of the Aorta in Computed Tomography Angiography via Context-Aware Shifted Window Self-Attention},
  author={Imran, Muhammad and Krebs, Jonathan R and Gopu, Veera Rajasekhar Reddy and Fazzone, Brian and Sivaraman, Vishal Balaji and Kumar, Amarjeet and Viscardi, Chelsea and Heithaus, Robert Evans and Shickel, Benjamin and Zhou, Yuyin and others},
  journal={arXiv preprint arXiv:2401.13049},
  year={2024}
}
2. @article{jiang2024microsegnet,
  title={MicroSegNet: a deep learning approach for prostate segmentation on micro-ultrasound images},
  author={Jiang, Hongxu and Imran, Muhammad and Muralidharan, Preethika and Patel, Anjali and Pensa, Jake and Liang, Muxuan and Benidir, Tarik and Grajo, Joseph R and Joseph, Jason P and Terry, Russell and others},
  journal={Computerized Medical Imaging and Graphics},
  volume={112},
  pages={102326},
  year={2024},
  publisher={Elsevier}
}
