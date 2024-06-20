<div align="center"><h1><center><u>Implementing a 3D Segmentation Model for Training</u></center></h1></div> <br>

This guide outlines the steps to implement a straightforward 3D segmentation model for training purposes. Follow the instructions below:

1. **Clone the Repository:**
   
   - Open your terminal or command prompt.
     
   - Use the following command to clone the repository:
     
     ```
     git clone git@github.com:ImranNust/AortaSeg24.git
     ```

3. **Navigate to the Training Directory:**

   - Change your working directory to the training folder within the cloned repository:
     
    ```
    cd AortaSeg24/training/
    ```

You are now ready to proceed with training your 3D segmentation model! 🚀


<h2><center>Directory Structure</center></h2>

It is assumed that you are in the training folder, and the training folder has the following files and folders inside it!

---
```
training/
├── data/                                  # Folder containing the training data
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
├── run_train.sh                           # Bash script to execute training process (likely calls train.py)
├── train.py                               # Python script containing the core training logic
├── utils/                                 # Folder containing utility functions used during training
|   ├── init.py                            # Empty file to mark utils as a Python package
|   ├── dataset.py                         # Python script containing functions for loading and processing data
|   └── trainer_script.py                  # Python script containing functions for training the model
└── ...                                    # Other project files (requirements.txt, etc.)
```
**Note:**
- The `data/` folder stores your training data, containing the corresponding images and segmentation masks.
- The `training/` folder holds the scripts used for training the model and the data.

---

<h2><center>Running Training</center></h2>
To train the model, run the following command:

```
./run_train.sh
```

This script will:

1. Create a virtual environment named "AortaSeg24".
2. Activate the environment.
3. Assumes all the required dependencies are already installed.
   
**If you encounter issues with the environment creation, you can:**

1. Create your own virtual environment.
2. Install the following dependencies listed in the "Dependencies" section.
3. Run the `./run_train.sh` script again.

<h3><center>Dependencies</center></h3>

The model was trained using the following Python libraries and versions:

<div align=center>
   
![Python: 3.10](https://img.shields.io/badge/3.10-Green?style=flat&logo=python&label=Python&labelColor=red&color=green) &emsp;
![PyTorch: 2.0.0+cu117](https://img.shields.io/badge/2.0.0%2Bcu117-violet?style=plastic&logo=pytorch&label=pytorch&labelColor=blue&color=green) &emsp; 
![Torchvision: 0.15.1+cu117](https://img.shields.io/badge/0.15.1%2Bcu117-violet?style=plastic&logo=pytorch&label=Torchvision&labelColor=blue&color=%23EE4C2C) &emsp; 
![MONAI: 1.3.0](https://img.shields.io/badge/1.3.0-violet?style=plastic&logo=monzo&label=monai&labelColor=blue&color=%23EE4C2C) &emsp; 
![SimpleITK: 2.3.1](https://img.shields.io/badge/2.3.1-violet?style=plastic&logo=Sanity&label=SimpleITK&labelColor=blue&color=%23EE4C2C) &emsp; 

![Pandas: 2.2.1](https://img.shields.io/badge/2.2.1-violet?style=plastic&logo=pandas&label=Pandas&labelColor=blue&color=%23EE4C2C) &emsp; 
![TQDM: 4.66.2](https://img.shields.io/badge/4.66.2-violet?style=plastic&logo=tqdm&label=TQDM&labelColor=blue&color=%23EE4C2C) &emsp; 
![Typing Extensions: 4.11.0](https://img.shields.io/badge/4.11.0-violet?style=plastic&logo=Git%20Extensions&label=Typing%20Extensions&labelColor=blue&color=%23EE4C2C) &emsp; 
</div>

**Note:** These are the specific versions used during training. Compatibility with newer versions might be possible, but it's recommended to stick to the listed versions for consistency.

<h3><center>Parameters Details</center></h3>

You can split the data into training and validation sets. For example, if you have 50 images and you want to use 40 images for training and 10 images for validation, you need to change the **--num_train_images** parameter to 40 either in the _run_train.sh_ file or _trian.py_ scripts. This will automatically reserve 40 images for training and 10 for validation. If, for some reason, you don't want to reserve any images for validation, you'll need to comment out the lines involving validation.

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
