## Getting the data
1. Download the dataset from https://universe.roboflow.com/custommodelsforbusiness/actors-face-recognition-ohq6j/dataset/1/download
in YOLOv9 format
2. extract the images to the images/folder
3. Use the test set

## Running the code
1. Create the conda environment using the ```environment.yml``` file and install the required libraries using poetry ```poetry install```
2. To do hyperparameter search run ```python find_parameters.py``` 
3. To run the cluster classification with the parameters located in ```model_parameters.json``` run ```process.py```

## Running the webapp
1. ```cd``` into ```webapp/```
2. run ```python app.py```