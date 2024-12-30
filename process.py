import faiss
import glob
import json
import numpy as np
from ultralytics import YOLO

from config import *
from utils import compute_distances, create_clusters
import shutil


def extract_img_name(path):
    return path.split("\\")[-1].split("/")[-1].split(".")[0]

def replace_file_paths(file_dict, original_images):
    updated_dict = {}
    for key, paths in file_dict.items():
        updated_paths = []
        for path in paths:
            # Extract the base filename and its extension
            base_name = path.split('\\')[-1]  # Get the filename with extension
            filename, extension = base_name.rsplit('.', 1)  # Separate filename and extension
            
            # Check if the filename starts with any of the original images
            for original in original_images:
                if filename.startswith(original):
                    # Replace with the original image name, retaining the extension
                    updated_paths.append(f"{original}.{extension}")
                    break
            else:
                # Keep the original path if no match is found
                updated_paths.append(path)
        
        updated_dict[key] = list(set(updated_paths))
    return updated_dict

#Predict initial human crops
model = YOLO(MODEL_NAME)
shutil.rmtree('humans/', ignore_errors=True)
results = model.predict(IMG_GLOB_PATH,classes=[0],save=True,
                        save_crop=True,project='humans',name="predict",
                        )
#Load tuned model:
with open('model_parameters.json') as f:
    model_parameters = json.load(f)

results_on_crops = model.predict(model_parameters['crops_path'], classes=[0], embed=[model_parameters['layer']], project='discard')

#Create image clusters:
embeddings = np.array(results_on_crops)
faiss.normalize_L2(embeddings)
distance = compute_distances(embeddings, model_parameters['distance_type'])
clusters = create_clusters(distance,glob.glob(f"{model_parameters['crops_path']}/*.jpg"),model_parameters['clustering_threshold'])
images_list = glob.glob(f'{CROPS_PATH}/*.jpg')
images_list = [extract_img_name(img) for img in images_list]


# Replace paths in the dictionary
updated_file_dict = replace_file_paths(clusters, images_list)

# Track seen values and their associated keys
seen_values = {}
keys_to_remove = set()

for key, value in updated_file_dict.items():
    if value in seen_values.values():
        # If the value is already seen, mark the current key for removal
        keys_to_remove.add(key)
    else:
        # Otherwise, add the value to the seen set
        seen_values[key] = value

# Create a new dictionary with duplicate values removed
updated_file_dict = {int(key): list(value) for key, value in updated_file_dict.items() if key not in keys_to_remove}

#Export predicted clusters
with open('webapp/predicted_clusters.json', 'w') as f:
    json.dump(updated_file_dict, f, indent=4)