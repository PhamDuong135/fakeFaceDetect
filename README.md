# fakeFaceDetect

## Set up for Python
1. Install Python or run it on virtual env (version 3.8) : ([Download](https://www.python.org))

2. Install all Python packages in file "Requirement.txt"


## Set up C compiler for Desktop App:
1. Install Visual Studio Community Ver
2. Select Desktop development with C++ in Visual Studio Installer

## Create Directory
1. Create a Dataset folder containing All, DataCollect, Fake, Real and SplitData subfolders to store data
![Dataset_folder_tree](https://github.com/user-attachments/assets/35c96d2e-ab3e-427c-bd64-2139aea934e5)


## Collect Data
1. To Collect Real Face Data: change classID to 0
2. To Collect Fake Face Data: change classID to 1
3. Run dataCollection.py
4. Copy all the data collected in the real and fake folders with a balanced ratio into 'All' folder

## Split Data
1. Run SplitData.py to split data for model training

## Model Training
1. Copy and rename file data.yaml in SplitData folder to dataOffline.yaml
2. Change the path in the dataOffline.yaml file to the absolute path of the SplitData folder
3. Change train, val, test path by: train/images, val: val/images, test: test/images
4. Run train.py

## Prepare Model
1. Copy the best.pt file in the runs/detect/train/weights directory to the models directory
2. Rename file best.pt (for example: n_version_1)
3. Change path to model in main.py file [model = YOLO("../models/path_to_n_version_1.pt")]

## Run Application
1. Run file main.py to start application
