import numpy as np
import pandas as pd
import argparse
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctrl_type", type=str, default="CTRL")
    parser.add_argument("--stain_type", type=str, default="All")
    parser.add_argument("--rett_type", type=str, default="HPS3042")
    parser.add_argument("--data_path", type=str, default="/home/acd13264yb/DDDog/Rettsyndrome/Classification/Datasets")
    parser.add_argument("--save_path", type=str, default="/home/acd13264yb/DDDog/Rettsyndrome/Biomarkertables/features_chromatin")
    args = parser.parse_args()
data_path = args.data_path
save_path = args.save_path
ctrl_type = args.ctrl_type
stain_type = args.stain_type
rett_type = args.rett_type

print("🚀 1. Load RETT Images and Compute metrics", flush=True)
total_results = {}
if ctrl_type == "CTRL":
    image_path = f"{ctrl_type}_{stain_type}"
elif ctrl_type == "RETT":
    image_path = f"{ctrl_type}_{rett_type}_{stain_type}"
data = np.load(f"{data_path}/{image_path}.npy", allow_pickle=True)
print(f"Load {image_path} data.shape {data.shape}", flush=True)

results = []

for i in range(len(data)):
    if i % 400 == 0: print(f"Process {i}/{len(data)}", flush=True)
    
    image = data[i, :, :, 0]

    # max_eigenvalue = compute_largest_eigenvalue(image, sigma=1, pad_width=10)
    cc_mask = thre_h_watershed(image, min_distance=5, max_area=300)
    metrics = calculate_quantitative_metrics(image, cc_mask)

    # 将metrics字典转换为一行
    result_row = {
        'chromatin_num': metrics['chromatin_num'],
        'nuclear_area': metrics['nuclear_area'],
        'chromatin_area': metrics['chromatin_area'],
        'RCA-S': metrics['RCA-S'],
        'RCA-M': metrics['RCA-M'],
        'nuclear_intensity': metrics['nuclear_intensity'],
        'chromatin_intensity': metrics['chromatin_intensity'],
        'RCI-S': metrics['RCI-S'],
        'RCI-M': metrics['RCI-M'],
    }
    results.append(result_row)

# 创建DataFrame
df = pd.DataFrame(results)
total_results[image_path] = df

# 保存DataFrame到文件
df.to_csv(f"{save_path}/{image_path}.csv", index=False)
print(f"🔥 SAVE to {save_path}/{image_path}.csv", flush=True)