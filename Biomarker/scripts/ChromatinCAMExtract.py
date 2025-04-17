import numpy as np
import pandas as pd
import argparse
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctrl_type", type=str, default="CTRL")
    parser.add_argument("--stain_type", type=str, default="All")
    parser.add_argument("--rett_type", type=str, default="HP9999")
    parser.add_argument("--data_path", type=str, default="/home/acd13264yb/DDDog/Rettsyndrome/Classification/results_cam")
    parser.add_argument("--save_path", type=str, default="/home/acd13264yb/DDDog/Rettsyndrome/Biomarkertables/features_chromatinCAM")
    args = parser.parse_args()
data_path = args.data_path
save_path = args.save_path
ctrl_type = args.ctrl_type
stain_type = args.stain_type
rett_type = args.rett_type

print("🚀 1. Load RETT Images and Compute metrics", flush=True)
total_results = {}
image_path = f"{ctrl_type}_{stain_type}"
cam = np.load(f"{data_path}/{image_path}_Resnet10_noavg_ScoreCAM/{image_path}_Resnet10_noavg_ScoreCAM_cam.npy", allow_pickle=True)
print(f"{ctrl_type}_{stain_type} cam", cam.shape, flush=True)
img = np.load(f"{data_path}/{image_path}_Resnet10_noavg_ScoreCAM/{image_path}_Resnet10_noavg_ScoreCAM_img.npy", allow_pickle=True)
print(f"{ctrl_type}_{stain_type} img", img.shape, flush=True)
if len(img)!=len(cam): 
    print("ERRO len(img)!=len(cam) break")

results = []
for n in range(len(img)):
    if n % 400 == 0: print(f"Process {n}/{len(img)}", flush=True)

    image = img[n,:,:,0]
    camm = cam[n,:,:]
    camm = camm > threshold_otsu(camm)

    max_eigenvalue = compute_largest_eigenvalue(image, sigma=1, pad_width=25)
    cc_mask = apply_h_watershed(max_eigenvalue, min_distance=5) * camm
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
print(f"🔥 SAVE to{save_path}/{image_path}.csv", flush=True)

