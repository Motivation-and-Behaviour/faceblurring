# Face Blurring

## Description

This script is designed to support the KidVision research project.
It is designed to work with the Brinno TLC130, although it should also support other Brinno cameras.
It will blur faces in the video files, and save the blurred videos to a new folder.
The original files are destroyed, per ethics requirements.

# Requirements
You need Conda installed to build the environment. 
You can get it [here](https://docs.conda.io/en/latest/miniconda.html).

To run on GPU (strongly recommended if deploying in the field), you will also need to have CUDA and CUDNN installed.

# Install
1. Clone the repo, either using Git or by downloading the zip.
2. Use `cd` to move to the directory with the repo.
3. Create the Conda environment. In Terminal (or Conda Prompt) run:

```
conda env create -f environment.yml
```

# To Run
1. Activate the environment. 
In Terminal (or Conda Prompt) run:

```
conda activate faceblurring
```

2. Run the script. 
In Terminal (or Conda Prompt) run:

```
python face_blurring.py
```

3. Follow the prompts to pick the participant ID and the location of the video files.