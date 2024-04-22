<h1>Multi-Class Segmentation of Aortic Branches and Zones in Computed Tomography Angiography</h1>

Welcome to the **AortaSeg24** challenge, where innovation meets collaboration in the pursuit of medical imaging excellence.

![Segmentation Demo](./assets/segmentation_demo.gif)


---

<h2>About the Challenge</h2>

This challenge focuses on proposing deep learning models for the segmentation of the Aorta, its branches, and zones. We provide an [example script](https://github.com/ImranNust/AortaSeg24_Duplicate/tree/main/training) utilizing [SwinUNETR](https://arxiv.org/abs/2201.01266) to guide you in developing your training script and training a module. Additionally, [sample validation code](https://github.com/ImranNust/AortaSeg24_Duplicate/tree/main/validation) is available to assess your proposed model's performance. Finally, a container submission script is offered to facilitate the submission process to the Grand Challenge. Feel free to modify this script to suit your specific needs.

---

<h2><u><center> Dataset Details </center></u></h2>

### Input Image Specifications
- **Type**: 3D CTA Volume
- **Axial Dimensions**: 
  - **Minimum**: $389\times389$ pixels
  - **Maximum**: $516\times516$ pixels
  - **Average**: $450\times450$ pixels
- **Voxel Resolution**: Isotropic, uniformly set to $\(1mm \times 1mm \times 1mm\)$.
- **Axial Slice Count**: 
  - **Range**: 578 to 801 slices
  - **Average**: 695 slices

### Output Segmentation Specifications
The output is a 3D segmentation model mirroring the input image characteristics, with the addition of 23 segmentation classes detailed below.

#### Segmentation Classes

![Details of the Segmentation Labels](assets/Segmentation_Details.PNG)

### Accessing the Dataset

To access the dataset, please participate in the **[AortaSeg24 Challenge](https://aortaseg24.grand-challenge.org/)** hosted on the Grand Challenge platform. Begin by visiting the challenge page and proceed to complete the **[Data Agreement Form](https://aortaseg24.grand-challenge.org/dataset-access-information/)**. Upon submission and approval, you will gain full access to the dataset.

---
