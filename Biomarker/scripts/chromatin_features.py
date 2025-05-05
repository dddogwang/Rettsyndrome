import cv2
import numpy as np
from scipy.spatial import distance
from skimage import measure
from skimage.measure import regionprops_table


# -----------------------------------------------------------------------------#
# 基础指标
# -----------------------------------------------------------------------------#
def chromatin_centroid(cc_mask: np.ndarray) -> np.ndarray:
    """返回 Nx2 质心坐标 (row, col)，整数化便于索引。"""
    centroids = np.array([prop.centroid for prop in measure.regionprops(cc_mask)])
    return np.round(centroids).astype(int)


def compute_chromatin_number(cc_mask: np.ndarray) -> int:
    """染色中心数量（标签最大值，假设标签从 1 开始）。"""
    return int(cc_mask.max())


def compute_chromatin_area(
    nucleus_image: np.ndarray, cc_mask: np.ndarray, num_cc: int
) -> dict[str, float]:
    """面积相关指标：核面积、平均颗粒面积、相对面积 (RCA-S/M)。"""
    metrics = {}

    # 1. 核面积（非零像素计数）
    nuclear_area = np.count_nonzero(nucleus_image)
    metrics["nuclear_area"] = float(nuclear_area)

    # 2. 每颗粒面积列表
    cc_areas = regionprops_table(cc_mask, properties=["area"])["area"]
    metrics["chromatin_area"] = float(np.mean(cc_areas)) if cc_areas.size else np.nan

    # 3. 相对面积
    with np.errstate(divide="ignore", invalid="ignore"):
        metrics["RCA-S"] = float(np.sum(cc_areas) / nuclear_area)
        metrics["RCA-M"] = float(np.mean(cc_areas) / nuclear_area)

    return metrics


def compute_chromatin_intensity(
    nucleus_image: np.ndarray, cc_mask: np.ndarray, num_cc: int
) -> dict[str, float]:
    """强度相关指标：核平均灰度、颗粒平均灰度、相对强度 (RCI-S/M)。"""
    metrics = {}

    nuclear_pixels = nucleus_image[nucleus_image > 0]
    nuclear_intensity = float(np.mean(nuclear_pixels)) if nuclear_pixels.size else np.nan
    metrics["nuclear_intensity"] = nuclear_intensity

    cc_intensities = regionprops_table(
        cc_mask, intensity_image=nucleus_image, properties=["mean_intensity"]
    )["mean_intensity"]
    metrics["chromatin_intensity"] = (
        float(np.mean(cc_intensities)) if cc_intensities.size else np.nan
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        metrics["RCI-S"] = float(np.sum(cc_intensities) / nuclear_intensity)
        metrics["RCI-M"] = float(np.mean(cc_intensities) / nuclear_intensity)

    return metrics


def compute_chromatin_shape(cc_mask: np.ndarray, num_cc: int) -> float:
    """平均长/短轴比；若无合法颗粒返回 NaN。"""
    axis_ratios = [
        r.major_axis_length / r.minor_axis_length
        for r in measure.regionprops(cc_mask)
        if r.minor_axis_length > 0
    ]
    return float(np.mean(axis_ratios)) if axis_ratios else np.nan

# -----------------------------------------------------------------------------#
# 分布特征
# -----------------------------------------------------------------------------#

def compute_chromatin_distribution(nucleus_image: np.ndarray, centroids: np.ndarray,
                                   num_parts: int = 5,
                                   num_per_area: int = 250) -> np.ndarray:

    _, thresh = cv2.threshold(nucleus_image, 0, 1, cv2.THRESH_BINARY)
    mask = thresh.copy()
    h, w = thresh.shape[:2]
    mask_gap = 255//num_parts
    
    for i in range(num_parts, 0, -1):
        scale = i/num_parts
        new_h, new_w = int(h * scale), int(w * scale)
        d_h, d_w = (h - new_h) // 2, (w - new_w) // 2
        resized = cv2.resize(thresh, (new_w, new_h),
                             interpolation=cv2.INTER_NEAREST)
        copy = np.zeros_like(thresh)
        copy[d_h:d_h + new_h, d_w:d_w + new_w] = resized*2
        copy[copy == 0] = 1
        mask *= copy
    mask[mask == 1] = mask_gap
    
    part_area = np.zeros(num_parts)
    part_centroids_num = np.zeros(num_parts)
    part_centroids_num_per_area = np.zeros(num_parts)
    
    for i in range(num_parts):
        mask[mask==2**(i+1)] = mask_gap*(i+1)
        part_area[i] += np.sum(mask==mask_gap*(i+1))

    for i in range(len(centroids)):
        part_num = mask[centroids[i, 0], centroids[i, 1]]//mask_gap
        layer_idx = int(part_num - 1)
        if 0 <= layer_idx < num_parts: part_centroids_num[layer_idx] += 1          # 防止负索引回绕

    for i in range(num_parts):
        part_centroids_num_per_area[i] = (
            part_centroids_num[i]*num_per_area/part_area[i]
            if part_area[i] else 0.0
        )
    
    return part_centroids_num_per_area


# -----------------------------------------------------------------------------#
# H3K27ac / CTCF 共定位
# -----------------------------------------------------------------------------#
def src2tgt_mindist(src_centroids: np.ndarray,
                    tgt_centroids: np.ndarray) -> list[float]:
    """计算 src 粒子到 tgt 粒子最近距离 (N×M)"""
    if len(src_centroids) == 0:
        return []
    if len(tgt_centroids) == 0:
        return [np.nan] * len(src_centroids)

    # 向量化计算距离矩阵 (N × M)
    dists = distance.cdist(src_centroids, tgt_centroids, metric="euclidean")
    return dists.min(axis=1).tolist()

def src2tgt_circle_radius(src_centroids: np.ndarray,
                          tgt_centroids: np.ndarray,
                          maxradii: int = 500,
                          step: int = 1) -> dict[int, float]:
    """
    计算 src 粒子在不同半径范围内的 tgt 粒子数量 (NxM)
    """
    radii = np.arange(0, maxradii, step, dtype=int)

    # 边界情况
    if len(src_centroids) == 0:
        return {r: np.nan for r in radii}
    if len(tgt_centroids) == 0:
        return {r: 0.0 for r in radii}

    dists = distance.cdist(src_centroids, tgt_centroids, metric="euclidean")  # N×M

    avg_counts = {}
    for r in radii:
        avg_counts[r] = float(np.mean((dists <= r).sum(axis=1)))
    return avg_counts

# -----------------------------------------------------------------------------#
# 汇总封装
# -----------------------------------------------------------------------------#
def compute_chromatin_metrics(
    nucleus_image: np.ndarray, cc_mask: np.ndarray
) -> dict[str, float]:
    """
    单核图像的所有 chromatin 指标。
    返回 keys:
      chromatin_num, nuclear_area, chromatin_area, RCA-S, RCA-M,
      nuclear_intensity, chromatin_intensity, RCI-S, RCI-M,
      chromatin_shape, chromatin_distribution_part1-5
    """
    nucleus_image = nucleus_image.astype(np.float32)
    cc_mask = cc_mask.astype(np.uint16)

    metrics: dict[str, float] = {}

    # 数量
    num_cc = compute_chromatin_number(cc_mask)
    metrics["chromatin_num"] = num_cc

    # 面积 & 强度
    metrics.update(compute_chromatin_area(nucleus_image, cc_mask, num_cc))
    metrics.update(compute_chromatin_intensity(nucleus_image, cc_mask, num_cc))

    # 形态
    metrics["chromatin_shape"] = compute_chromatin_shape(cc_mask, num_cc)

    # 分布
    centroids = chromatin_centroid(cc_mask)
    distr = compute_chromatin_distribution(nucleus_image, centroids)
    for i, v in enumerate(distr):
        metrics[f"chromatin_distribution_part{5-i}"] = v

    return metrics


def compute_co_location_metrics(
    H3K27ac_mask: np.ndarray, CTCF_mask: np.ndarray
) -> dict[str, object]:
    """
    H3K27ac 与 CTCF 共定位指标：
      - h3k27ac2ctcf_mindist  (list)
      - h3k27ac2ctcf_radius  (dict[radius]->avg count)
      - ctcf2h3k27ac_mindist
      - ctcf2h3k27ac_radius
    """
    H3_centroids = chromatin_centroid(H3K27ac_mask)
    CTCF_centroids = chromatin_centroid(CTCF_mask)

    metrics: dict[str, object] = {}

    metrics["h3k27ac2ctcf_mindist"] = src2tgt_mindist(H3_centroids, CTCF_centroids)
    metrics["h3k27ac2ctcf_radius"] = src2tgt_circle_radius(H3_centroids, CTCF_centroids)
    metrics["ctcf2h3k27ac_mindist"] = src2tgt_mindist(CTCF_centroids, H3_centroids)
    metrics["ctcf2h3k27ac_radius"] = src2tgt_circle_radius(CTCF_centroids, H3_centroids)

    return metrics