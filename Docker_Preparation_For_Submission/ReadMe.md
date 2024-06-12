<div align=center>
    <h1>Docker Preparation for Challenge Submission </h1>

This document outlines the steps for creating a Docker container for challenge submission.

---

</div>

**Disclaimer:**
This is not an official guide for preparing Docker containers and submissions to the Grand Challenge platform. It serves as a supplementary tool. For comprehensive instructions, please refer to the Grand Challenge official documentation.

---

## Directory Structure

Before we begin, let's familiarize ourselves with the directory structure required for our Docker setup:


```
DockerExample/
├─ resources/                         # Directory for resources such as your saved model and other necessary utilities
|  ├─ your_best_model.pth             # Your trained model file
|  └── ....                           # Additional dependencies for predictions
|
├── Dockerfile                        # Dockerfile script to build the Docker image (Don't change it if you are not an expert!)
├── inference.py                      # Python script for generating output segmentations (Modify it as per your need)
├── requirements.txt                  # List of Python packages required for execution        
├── save.sh                           # Shell script to package the Docker image into a .tar.gz file
└── test.sh                           # Shell script for local testing of the Docker image
```

---

## Docker Image Creation
Ensure that [Docker](https://www.docker.com/) is installed on your local machine and you are familiar with its basic operations. If you’re new to Docker, consider watching this  [introductory video](https://www.youtube.com/watch?v=0UG2x2iWerk).

### Steps to Build Your Docker Image:
1. Launch the Docker application.
2. Open Visual Studio Code or your preferred code editor.
3. Navigate to the directory where your final files are located:
    - resources/
    - inference.py
    - save.sh
    - test.sh
    - requirements.txt
    - Dockerfile
4. Execute the following command in your terminal to build the Docker image:
```
docker build -t name_of_your_docker_image .
```
This command will compile the Docker image using the specifications in your Dockerfile.

5. To package your Docker image into a `.tar.gz` file, run the `save.sh` script:
```
chmod +x save.sh
./save.sh
```

Please note that in the `save.sh` file, the `container_tag` should be the same as the name of the container you used when you created it using `docker build` command. For example, in above line you created the docker with name `name_of_your_docker_image`; therefore, the the `containe_tag` should be `container_tag="name_of_your_docker_image"`


