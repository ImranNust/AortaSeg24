<h1><center><u>Training Tutorial</u></center></h1>

This guide provides instructions on implementing a straightforward 3D segmentation model for training purposes.
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
---
