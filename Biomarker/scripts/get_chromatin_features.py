import numpy as np
import pandas as pd
import argparse
from chromatin_segmentation import thre_h_watershed
from chromatin_features import compute_chromatin_metrics

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
data = np.load(f"{data_path}/{data_name}.npy", allow_pickle=True)
print(f"Load {data_name} data.shape {data.shape}", flush=True)

total_results = {}
results = []

for i in range(len(data)):
    if i % 100 == 0: print(f"Process {i}/{len(data)}", flush=True)
    
    image = data[i, :, :, 0]

    cc_mask = thre_h_watershed(image, min_distance=5, max_area=300)
    metrics = compute_chromatin_metrics(image, cc_mask)

    # 将metrics字典转换为一行
    result_row = {
        'chromatin_num': metrics['chromatin_num'],
        'chromatin_area': metrics['chromatin_area'],
        'chromatin_intensity': metrics['chromatin_intensity'],
        'chromatin_shape': metrics['chromatin_shape'],

        'nuclear_area': metrics['nuclear_area'],
        'nuclear_intensity': metrics['nuclear_intensity'],

        'chromatin_distribution_part1': metrics['chromatin_distribution_part1'],
        'chromatin_distribution_part2': metrics['chromatin_distribution_part2'],
        'chromatin_distribution_part3': metrics['chromatin_distribution_part3'],
        'chromatin_distribution_part4': metrics['chromatin_distribution_part4'],
        'chromatin_distribution_part5': metrics['chromatin_distribution_part5'],
        'RCA-S': metrics['RCA-S'],
        'RCA-M': metrics['RCA-M'],
        'RCI-S': metrics['RCI-S'],
        'RCI-M': metrics['RCI-M'],
    }
    results.append(result_row)

# 将所有结果转换为DataFrame
df = pd.DataFrame(results)
save_path = f"{save_path}/{data_name}_basic.csv"
df.to_csv(save_path, index=False)
print(f"🔥 SAVE to {save_path}", flush=True)