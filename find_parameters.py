import pandas as pd
from ultralytics import YOLO
import glob
import numpy as np
from utils import tune_parameters
import json
from config import *
import shutil


#Parameters
metrics_list = ['euclidean', 'cosine']
thresholds = np.linspace(0, 0.99, 51)
layers_list = range(0, 7)


if __name__ == "__main__":
    #Load ground truth.
    df_truth = pd.read_csv(GROUND_TRUTH_PATH)
    ground_truth_dict = df_truth.groupby('cluster')['picture'].apply(list).to_dict()

    #Initial prediction of human crops
    model = YOLO(MODEL_NAME)
    # Remove the path and all the files and folders inside 'humans/'
    shutil.rmtree('humans/', ignore_errors=True)
    model.predict('images/',classes=[0],save=True,
                        save_crop=True,project='humans',name="predict",
                        )
    result = tune_parameters(model, layers_list, metrics_list, thresholds, CROPS_PATH,ground_truth_dict)

    # Create a dictionary with custom keys for the desired fields
    export_data = {
        "distance_type": result[1],               
        "clustering_threshold": float(result[2]), 
        "layer": result[3],
        'model_name': MODEL_NAME,
        'crops_path': CROPS_PATH,

    }

    # Write dictionary to a JSON file
    with open("model_parameters.json", "w") as file:
        json.dump(export_data, file, indent=4)

    print("Model parameters saved to model_parameters.json")