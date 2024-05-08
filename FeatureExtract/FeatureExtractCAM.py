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
    parser.add_argument("--stain_type", type=str, default="All")
    parser.add_argument("--cam_type", type=str, default="GradCAM")
    parser.add_argument("--image_path", type=str, default="/home/acd13264yb/DDDog/Rettsyndrome/Classification/Datasets")
    parser.add_argument("--save_path", type=str, default="/home/acd13264yb/DDDog/Rettsyndrome/Profilling")
    args = parser.parse_args()
ctrl_type = args.ctrl_type
stain_type = args.stain_type
cam_type = args.cam_type
image_path = args.image_path
save_path = args.save_path
loadname = f"{ctrl_type}_{stain_type}_Resnet10_noavg_{cam_type}"
print(f"loadname: {loadname}")

print("🚀 1. Load Images and Compute Masks", flush=True)
img_all = np.load(f"../Classification/results_cam/{loadname}/{loadname}_img.npy",allow_pickle=True)
cam_all = np.load(f"../Classification/results_cam/{loadname}/{loadname}_cam.npy",allow_pickle=True)
print(f"img all shape: {img_all.shape}", flush=True)
print(f"cam all shape: {cam_all.shape}", flush=True)

mask_all = []
for img in img_all:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)
    mask_all.append(thresh)
mask_all = np.array(mask_all).astype(np.int32)
print(f"mask all shape: {mask_all.shape}", flush=True)
print("done", flush=True)
print("##########################################################", flush=True)

print("🚀 2. Compute and Save nuclei features", flush=True)


# Init DataFrame to save
features_all = pd.DataFrame()
# Compute and Save nuclei features
for i in tqdm(range(len(cam_all))):
    features = hf.compute_nuclei_features(im_label=mask_all[i], im_nuclei=cam_all[i,:,:])
    features["Label"] = i
    # Add new DataFrame to DataFrame 
    features_all = pd.concat([features_all, features], ignore_index=True)
# Save DataFrame as CSV
features_name = f'features_{loadname}.csv'
features_all.to_csv(features_name, index=False)
print(f"🔥 Save features_all as {features_name}", flush=True)
print("done", flush=True)
print("##########################################################", flush=True)