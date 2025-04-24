from skimage import io, feature, filters, transform, measure
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours

def is_close(point, other_points, threshold=5):
    """检查是否有任何点在阈值范围内"""
    for other in other_points:
        if np.linalg.norm(np.array(point) - np.array(other)) <= threshold:
            return True
    return False

def filter_contours_by_proximity(contours, nuclear_contours, proximity=5):
    """过滤掉靠近核心轮廓的轮廓"""
    new_contours = []
    # 展平 nuclear_contours 中的所有点
    all_nuclear_points = [point for contour in nuclear_contours for point in contour]

    for contour in contours:
        # 检查轮廓中的任何点是否靠近核心轮廓的点
        if not any(is_close(point, all_nuclear_points, proximity) for point in contour):
            new_contours.append(contour)

    return new_contours

def compute_largest_eigenvalue(image, sigma=1, pad_width=10):
    nuclear = (image!=0).astype(np.uint8)
    nuclear_scaled = transform.rescale(nuclear, (500-pad_width*2)/500)
    nuclear_padded = np.pad(nuclear_scaled, pad_width=pad_width, mode='constant', constant_values=0)

    # 计算结构张量
    result = feature.structure_tensor(image, sigma=sigma, order='rc')
    # 从结构张量中获取特征值
    eigenvalues = feature.structure_tensor_eigenvalues(result)
    # 返回每个点的最大特征值
    return np.max(eigenvalues, axis=0)*nuclear_padded

# 使用distance_transform_edt
def apply_h_watershed(image, min_distance=5):
    mask = image > filters.threshold_otsu(image[image > 0])
    # 计算距离变换
    distance = distance_transform_edt(mask)
    # 在距离图中找到峰值
    local_maxi = feature.peak_local_max(distance, min_distance=min_distance, labels=mask)
    # 将峰值的坐标转换为标记矩阵
    if len(local_maxi)<=255:
        markers = np.zeros_like(image, dtype=np.uint8)
    else:
#         print("len(local_maxi)>255")
        markers = np.zeros_like(image, dtype=np.int32)
    for i, (row, col) in enumerate(local_maxi):
        markers[row, col] = i + 1
    # 执行分水岭分割
    labels_ws = watershed(-distance, markers, mask=mask)
    return labels_ws

def thre_h_watershed(image, ratio=1, min_distance=5, classes=4, max_area=None):
    
#     # Otsu 阈值化
#     thre = threshold_otsu(image[image>0])
#     binary_image = image > thre * ratio

    # 使用多级 Otsu 阈值化
    thresholds = filters.threshold_multiotsu(image[image > 0], classes=classes)
    thre = thresholds[-1]
    binary_image = image > thre * ratio
    
    # Compute the distance transform
    distance = ndi.distance_transform_edt(binary_image)

    # Find local maxima
    local_maxi = feature.peak_local_max(distance, min_distance=min_distance, labels=binary_image)

    # Marker labeling
    if len(local_maxi)<=255:
        markers = np.zeros_like(image, dtype=np.uint8)
    else:
        markers = np.zeros_like(image, dtype=np.int32)
    for i, (row, col) in enumerate(local_maxi):
        markers[row, col] = i + 1
        
    # Apply watershed
    cc_mask = watershed(-distance, markers, mask=binary_image)
    
    # remove area >= max_area
    if max_area!=None:
        regions = measure.regionprops(cc_mask)
        # Create a mask for regions with area <= max_area
        mask = np.zeros_like(cc_mask, dtype=bool)
        for region in regions:
            if region.area <= max_area:
                mask[tuple(region.coords.T)] = True
        cc_mask = cc_mask * mask
    
    return cc_mask


def plot_contours(image, cc_mask):
    plt.imshow(image, cmap='gray')
    for i in range(cc_mask.max()):
        cc_contours = find_contours(cc_mask==i, level=0.5)
        for contour in cc_contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2)


def show_mask_and_metrics(ctrl_type, rett_type, chip_type, num, lr=False, home_path="../Classification"):
    
    if ctrl_type=="RETT":
        image_path = f"{ctrl_type}_{rett_type}_{chip_type}"
    elif ctrl_type=="CTRL":
        image_path = f"{ctrl_type}_{chip_type}"
    
    if not lr:
        image = np.load(f"{home_path}/Datasets/{image_path}.npy", allow_pickle=True)[num,:,:,0]
    elif lr:
        image = np.load(f"{home_path}/Datasets_LR/{image_path}.npy", allow_pickle=True)[num,:,:,0]
    
    cc_mask = thre_h_watershed(image, min_distance=5, max_area=1000)
    
    # Metrics particle (Heterochromatin)
    metrics = calculate_quantitative_metrics(image, cc_mask)
    print(f"💠 {ctrl_type}-{rett_type}-{chip_type} Calculate_quantitative_metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
        
    # plot mask
    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.imshow(cc_mask, cmap='gray')
    
    # plot contours
    plt.subplot(1,2,2)
    plt.imshow(image, cmap='gray')
    for i in range(cc_mask.max()):
        cc_contours = find_contours(cc_mask==i, level=0.5)
        for contour in cc_contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.show()
    
    return cc_mask