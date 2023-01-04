# Tensorflow Object Detection Setup

## Step By Step Guide

1. Clone the repository to your local machine;

2. Set up the Python Virtual environment  using Venv or Conda;

3. Run the **Image_Collenction** notebook;

4. Run the **Training** notebook (Here you have 2 options):
    - Run using your local machine;
    - Run using Google Collab (*Recommended*);

5. Run the **Detection** notebook to see object detection in real-time (*optional*)

## Setup Python Virtual Environment 

To avoid dependencies issues, this tutorial aims to help you how to create and configure a Virtual Environment. After cloning this repository to your machine, open terminal/comand line in the folder where you cloned the repository. If you have done the setup with Git correctly, follow the following steps.

#### Python 

**Create:**

``python -m venv tfod-env``

**Activate**

``source tfod/bin/activate # Linux``

``.\tfod-env\Scripts\activate # Windows ``

**Install dependencies and add virtual environment to the Python Kernel**

``python -m pip install --upgrade pip``

``pip install ipykernel``

``python -m ipykernel install --user --name=tfodj``

``pip install wget``

``pip install tensorflow``

``pip install opencv-python``

``pip install --upgrade pyqt5 lxml``


#### Conda 

1. Open terminal


2. Run the following command to create a new Conda Environment:

```bash
conda create --name <your-environment-name> python=3.9.0
```

3. If you have created your environment with no errors, activate your Conda Environment running the following command:

```bash
conda activate <your-environment-name>
```

4. Install the required libraries using *pip* like as follows:

```bash
pip install tensorflow pandas numpy matplotlib sklearn opencv-python pyqt5 lxml
```

5. Finally, run this command:

```bash
conda install -c conda-forge --update-deps --force-reinstall ipykernel -y
```

After that, you can open the *1.Image_Collection.ipynb* notebook and select the right kernel (i.e., select the Conda Environment you have created).