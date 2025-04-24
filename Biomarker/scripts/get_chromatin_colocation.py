import numpy as np
import pandas as pd
import argparse
from chromatin_segmentation import thre_h_watershed
from chromatin_features import compute_co_location_metrics
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--data_path", type=str, default="/home/acd13264yb/DDDog/disease_epigenome/results_RTT/Datasets_SR")
    parser.add_argument("--save_path", type=str, default="/home/acd13264yb/DDDog/disease_epigenome/results_RTT/Biomarker/results")
    args = parser.parse_args()

print("🚀 0. Load argparse paramaters", flush=True)
data_path = args.data_path
data_name = args.data_name
save_path = args.save_path
print(f"data_path: {data_path}", flush=True)
print(f"save_path: {save_path}", flush=True)
print(f"data_name: {data_name}", flush=True)

print("🚀 1. Load RETT Images and Compute metrics", flush=True)
H3K27ac_images = np.load(f"{data_path}/{data_name}_H3K27ac.npy", allow_pickle=True)[:,:,:,0]
CTCF_images = np.load(f"{data_path}/{data_name}_CTCF.npy", allow_pickle=True)[:,:,:,0]
print(f"Load {data_name}_H3K27ac data.shape {H3K27ac_images.shape}", flush=True)
print(f"Load {data_name}_CTCF data.shape {CTCF_images.shape}", flush=True)

metrics_co_localization = {
    "h3k27ac2ctcf_mindist":[],
    "h3k27ac2ctcf_radius": [],
    "ctcf2h3k27ac_mindist":[],
    "ctcf2h3k27ac_radius": []
}

if len(H3K27ac_images) != len(CTCF_images):
    raise ValueError(f"Length of H3K27ac images {len(H3K27ac_images)} and CTCF images {len(CTCF_images)} do not match.")

for num in range(len(H3K27ac_images)):
    if num % 400 == 0: print(f"Process {num}/{len(H3K27ac_images)}", flush=True)

    H3K27ac_mask = thre_h_watershed(H3K27ac_images[num], min_distance=5, max_area=300)
    CTCF_mask = thre_h_watershed(CTCF_images[num], min_distance=5, max_area=300)
    
    metrics = compute_co_location_metrics(H3K27ac_mask, CTCF_mask)
    metrics_co_localization["h3k27ac2ctcf_mindist"].append(metrics["h3k27ac2ctcf_mindist"])
    metrics_co_localization["h3k27ac2ctcf_radius"].append(metrics["h3k27ac2ctcf_radius"])
    metrics_co_localization["ctcf2h3k27ac_mindist"].append(metrics["ctcf2h3k27ac_mindist"])
    metrics_co_localization["ctcf2h3k27ac_radius"].append(metrics["ctcf2h3k27ac_radius"])

save_path = f"{save_path}/{data_name}_metrics_co_localization.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(metrics_co_localization, f)
print(f"🔥 SAVE to {save_path}", flush=True)