from skimage import measure
import numpy as np
import cv2
from scipy.spatial import distance

def safe_mean(x):
    return np.nanmean(x) if len(x) > 0 else np.nan

def compute_chromatin_number(cc_mask):
    """
    计算染色中心的数量。
    """
    return np.max(cc_mask)

def compute_chromatin_area(nucleus_image, cc_mask, num_cc):
    """
    计算染色中心的面积。
    """
    metrics_areas = {}
    
    # 计算细胞核面积
    nuclear_area = np.sum(nucleus_image > 0)
    metrics_areas['nuclear_area'] = nuclear_area
    
    # 计算平均chromatin面积 (CA)
    cc_areas = [np.count_nonzero(cc_mask == i) for i in range(1, num_cc + 1)]
    metrics_areas['chromatin_area'] = np.mean(cc_areas) if cc_areas else np.nan

    # 计算相对(核)chromatin面积和 (RCA-S)
    metrics_areas['RCA-S'] = np.sum(cc_areas) / nuclear_area if nuclear_area > 0 else np.nan

    # 计算相对(核)chromatin面积平均 (RCA-M)
    metrics_areas['RCA-M'] = np.mean(cc_areas) / nuclear_area if cc_areas and nuclear_area > 0 else np.nan

    return metrics_areas

def compute_chromatin_intensity(nucleus_image, cc_mask, num_cc):
    """
    计算染色中心的强度。
    """
    metrics_intensity = {}
    
    # 计算细胞核强度
    nuclear_intensity = np.mean(nucleus_image[nucleus_image > 0])
    metrics_intensity['nuclear_intensity'] = nuclear_intensity
    
    # 计算平均chromatin强度 (CI)
    cc_intensities = [np.mean(nucleus_image[cc_mask == i]) for i in range(1, num_cc + 1) if np.any(cc_mask == i)]
    metrics_intensity['chromatin_intensity'] = np.mean(cc_intensities) if cc_intensities else np.nan

    # 计算相对(核)chromatin强度和 (RCI-S)
    metrics_intensity['RCI-S'] = np.sum(cc_intensities) / nuclear_intensity if nuclear_intensity > 0 else np.nan

    # 计算相对(核)chromatin强度平均 (RCI-M)
    metrics_intensity['RCI-M'] = np.mean(cc_intensities) / nuclear_intensity if cc_intensities and nuclear_intensity > 0 else np.nan

    return metrics_intensity

def compute_chromatin_shape(cc_mask):
    """
    计算染色中心的形状。
    """
    # 提取每个连通区域（排除背景0）
    all_cc_mask = [cc_mask == i for i in range(1, cc_mask.max())]
    all_cc_mask = np.array(all_cc_mask).astype(int)

    all_Axis_ratio = []

    for mask in all_cc_mask:
        regions = measure.regionprops(mask)
        if len(regions) == 1:
            region = regions[0]
            if region.area >= 1 and region.minor_axis_length > 0:
                axis_ratio = region.major_axis_length / region.minor_axis_length
                all_Axis_ratio.append(axis_ratio)

    return np.mean(all_Axis_ratio) if all_Axis_ratio else np.nan

def chromation_centroid(cc_mask):
    """
    计算染色中心的质心。
    """
    cc_props = measure.regionprops(cc_mask)
    centroids = np.array([prop.centroid for prop in cc_props])
    centroids = np.round(centroids).astype(int)
    return centroids

def compute_chromation_distribution(image, centroids, num_parts=5, num_per_area=250):
    _, thresh = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY)
    mask = thresh.copy()
    h, w = thresh.shape[:2]
    mask_gap = 255//num_parts
    
    for i in range(num_parts, 0, -1):
        scale = i/5
        new_h, new_w = int(h * scale), int(w * scale)
        d_h, d_w = (h - new_h) // 2, (w - new_w) // 2
        resized = cv2.resize(thresh, (new_w, new_h))
        copy = np.zeros_like(thresh)
        copy[d_h:d_h + new_h, d_w:d_w + new_w] = resized*2
        copy[copy == 0] = 1
        mask *= copy
        
    part_area = np.zeros(num_parts)
    part_centroids_num = np.zeros(num_parts)
    part_centroids_num_per_area = np.zeros(num_parts)
    
    for i in range(num_parts):
        mask[mask==2**(i+1)] = mask_gap*(i+1)
        part_area[i] += np.sum(mask==mask_gap*(i+1))

    for i in range(len(centroids)):
        part_num = mask[centroids[i, 0], centroids[i, 1]]//mask_gap
        part_centroids_num[part_num.astype(int)-1] += 1

    for i in range(num_parts):
        part_centroids_num_per_area[i] = part_centroids_num[i]*num_per_area/part_area[i]
    
    return part_centroids_num_per_area

def H3K27ac_centroids_ctcf_mindist(H3K27ac_centroids, CTCF_centroids):
    # 计算每个H3K27ac粒子到最近的CTCF粒子的距离
    nearest_distances = []
    for h3_centroid in H3K27ac_centroids:
        distances = distance.cdist([h3_centroid], CTCF_centroids, 'euclidean')
        nearest_distance = np.min(distances)
        nearest_distances.append(nearest_distance)
    
    return nearest_distances

def H3K27ac_cirlce_ctcf_radius(H3K27ac_centroids, CTCF_centroids, maxradii=50):
    # 设置要计算的不同半径范围
    radii = np.arange(0, maxradii, 1)  # 0到50像素，步长为5

    # 计算每个H3K27ac粒子不同半径范围内的CTCF粒子数量
    counts_per_radius = {radius: [] for radius in radii}

    for h3_centroid in H3K27ac_centroids:
        distances = distance.cdist([h3_centroid], CTCF_centroids, 'euclidean')[0]
        for radius in radii:
            count_within_radius = np.sum(distances <= radius)
            counts_per_radius[radius].append(count_within_radius)

    # 计算每个半径范围内的平均CTCF粒子数量
        average_counts_per_radius = {
        radius: np.mean(counts) if counts else np.nan
        for radius, counts in counts_per_radius.items()
    }
    
    return average_counts_per_radius 


def compute_chromatin_metrics(nucleus_image, cc_mask):
    """
    计算细胞核图像的量化指标。
    
    参数:
    nucleus_image: numpy.ndarray, 细胞核图像，灰度图
    cc_mask: numpy.ndarray, 染色中心的掩膜，二值图
    
    返回:
    metrics: dict, 包含所有量化指标的字典
    """
    nucleus_image = nucleus_image.astype(np.float32)
    cc_mask = cc_mask.astype(np.uint16)

    metrics = {}
    
    # 计算染色中心的数量
    num_cc = compute_chromatin_number(cc_mask)
    metrics['chromatin_num'] = num_cc
    
    # 计算细胞核面积和染色中心面积
    metrics.update(compute_chromatin_area(nucleus_image, cc_mask, num_cc))

    # 计算细胞核强度和染色中心强度
    metrics.update(compute_chromatin_intensity(nucleus_image, cc_mask, num_cc))
    
    # 计算染色中心的形状
    metrics['chromatin_shape'] = compute_chromatin_shape(cc_mask)
    
    # 计算染色中心的质心
    centroids = chromation_centroid(cc_mask)
    chromatin_distribution = compute_chromation_distribution(nucleus_image, centroids)
    metrics['chromatin_distribution_part1'] = chromatin_distribution[0]
    metrics['chromatin_distribution_part2'] = chromatin_distribution[1]
    metrics['chromatin_distribution_part3'] = chromatin_distribution[2]
    metrics['chromatin_distribution_part4'] = chromatin_distribution[3]
    metrics['chromatin_distribution_part5'] = chromatin_distribution[4]

    return metrics
 
def compute_co_location_metrics(H3K27ac_mask, CTCF_mask):
    """
    计算H3K27ac和CTCF粒子之间的共定位指标。
    
    参数:
    H3K27ac_mask: numpy.ndarray, H3K27ac粒子的掩膜
    CTCF_mask: numpy.ndarray, CTCF粒子的掩膜
    
    返回:
    metrics: dict, 包含所有共定位指标的字典
    """

    H3K27ac_centroids = chromation_centroid(H3K27ac_mask)
    CTCF_centroids = chromation_centroid(CTCF_mask)

    metrics = {}
    
    # 计算H3K27ac粒子到最近CTCF粒子的距离
    metrics['h3k27ac2ctcf_mindist'] = H3K27ac_centroids_ctcf_mindist(H3K27ac_centroids, CTCF_centroids)
    
    # 计算H3K27ac粒子在不同半径范围内的CTCF粒子数量
    metrics['h3k27ac2ctcf_radius'] = H3K27ac_cirlce_ctcf_radius(H3K27ac_centroids, CTCF_centroids)

    # 计算CTCF粒子到最近H3K27ac粒子的距离
    metrics['ctcf2h3k27ac_mindist'] = H3K27ac_centroids_ctcf_mindist(CTCF_centroids, H3K27ac_centroids)
    
    # 计算CTCF粒子在不同半径范围内的H3K27ac粒子数量
    metrics['ctcf2h3k27ac_radius'] = H3K27ac_cirlce_ctcf_radius(CTCF_centroids, H3K27ac_centroids)
    
    return metrics