import pandas as pd
from ultralytics import YOLO
import glob
import numpy as np
from utils import tune_parameters
import json
from config import *


#Parameters
metrics_list = ['euclidean', 'cosine']
thresholds = np.linspace(0, 0.5, 51)
layers_list = range(0, 7)


if __name__ == "__main__":
    #Load ground truth.
    df_truth = pd.read_excel('true_labels.xlsx')
    ground_truth_dict = df_truth.groupby('cluster')['foto'].apply(list).to_dict()

    #Initial prediction of human crops
    model = YOLO(MODEL_NAME)
    list_imgs = glob.glob(f"{CROPS_PATH}/*JPG")
    model.predict(IMG_PATH,classes=[0],save=True,
                        save_crop=True,project='humans',name="predict",
                        )
    result = tune_parameters(model, layers_list, metrics_list, thresholds, CROPS_PATH, list_imgs,ground_truth_dict)

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