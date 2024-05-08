print("🥁 PYTHON SCRIPT START", flush=True)
# 0. Import Libraries and Mount Drive
print("🚀 0. Import Libraries and Mount Drive", flush=True)
import histomicstk.features as hf
import numpy as np
import cv2, argparse
import pandas as pd
from tqdm import tqdm
from skimage.measure import label
from matplotlib import pyplot as plt
print("done", flush=True)
print("##########################################################", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctrl_type", type=str, default="CTRL")
    parser.add_argument("--image_path", type=str, default="/home/acd13264yb/DDDog/Rettsyndrome/Classification/Datasets")
    parser.add_argument("--save_path", type=str, default="/home/acd13264yb/DDDog/Rettsyndrome/Profilling")
    args = parser.parse_args()
ctrl_type = args.ctrl_type
image_path = args.image_path
save_path = args.save_path

print("🚀 1. Load Images and Compute Masks", flush=True)
stain_type = ["H3K27ac", "CTCF", "Dapi"]
img_all = np.load(f"{image_path}/{ctrl_type}_All.npy",allow_pickle=True)
mask_all = []

for img in img_all:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)
    mask_all.append(thresh)
mask_all = np.array(mask_all)
print(f"img all shape: {img_all.shape}", flush=True)
print(f"mask all shape: {mask_all.shape}", flush=True)
print("done", flush=True)
print("##########################################################", flush=True)

print("🚀 2. Compute and Save nuclei features", flush=True)
# Compute and Save nuclei features
for c in range(3):
    # Init DataFrame to save
    features_all = pd.DataFrame()
    for i in range(len(img_all)):
        features = hf.compute_nuclei_features(im_label=mask_all[i], im_nuclei=img_all[i,:,:,c])
        features["Label"] = i
        # Add new DataFrame to DataFrame 
        features_all = pd.concat([features_all, features], ignore_index=True)
    # Save DataFrame as CSV
    features_name = f'{save_path}/features_{ctrl_type}_{stain_type[c]}.csv'
    features_all.to_csv(features_name, index=False)
    print(f"🔥 Save features_all as {features_name}")
print("done", flush=True)
print("##########################################################", flush=True)