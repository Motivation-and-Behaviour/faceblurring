# Face Blurring with YOLO

**Adknowledgements**: The code borrows heavily from [YOLOFace](https://github.com/sthanhng/yoloface). I've just edited to fit my workflow.

# Install
1. Clone the repo, either using Git or by downloading the zip.
2. Use `cd` to move to the directory with the repo.
3. Create the Conda environment. In Terminal (or Conda Prompt) run:

`conda env create -f environment.yml`

4. Download the weight files. These are too large for GitHub so are stored on Google Drive. Download [yolov3-face.cfg](https://drive.google.com/file/d/1j0SVta521wNo6KwX-oAbSzh-0wKqO1nM/view?usp=sharing) and [yolov3-wider_16000.weights](https://drive.google.com/open?id=1lBNAgRyQQyWGFQjnu4-n6lEOYf7Nqiiw) and put both in the `weights` folder.

# To Run
1. Activate the environment. In Terminal (or Conda Prompt) run:

`conda activate faceblurring`

2. Start Jupyter (I prefer Lab, but it will run in Notebook too):

`jupyter lab`

3. Run the cells at the top. Change the `path_to_...` to point to the file or folder you want to run before running those cells.

## Outputs
The outputs go in the same directory as the original file/folder. It will append "_blurred" to the filename to prevent overwriting your original files.
